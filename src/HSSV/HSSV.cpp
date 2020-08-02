#include <HSSV.h>
#include <Octree.h>
#include <CPU/CpuBuilder.h>
#include <Defines.h>
#include <OctreeSerializer.h>
#include <HighResolutionTimer.h>

#include <GSCaps.h>

#include <Model.h>
#include <FunctionPrologue.h>
#include <createAdjacency.h>
#include <MathOps.h>

#include <CPU/CpuSidesDrawer.h>
#include <GPU/GpuSidesDrawer.h>

#include <iostream>
#include <fstream>
#include <iomanip>

HSSV::HSSV(vars::Vars& vars) : ShadowVolumes(vars)
{
}

HSSV::~HSSV()
{
	vars.erase("hssv.objects");
}

void HSSV::drawSides(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	resetMultiplicity();
	createAdjacency(vars);

	if (vars.getBool("hssv.args.buildTest"))
	{
		buildTest();
		exit(0);
	}

	getOctree();
	createSidesDrawer();

	vars.getReinterpret<SidesDrawerBase>("hssv.objects.sidesDrawer")->drawSides(projectionMatrix * viewMatrix, lightPosition);
}

void HSSV::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	createCapsDrawer();

	vars.get<GSCaps>("hssv.objects.capsDrawer")->drawCaps(lightPosition, viewMatrix, projectionMatrix);
}

void HSSV::drawUser(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{

}

void HSSV::getOctree()
{
	FUNCTION_PROLOGUE("hssv.objects", 
		"renderModel", 
		"maxMultiplicity",
		"hssv.args.buildCpu", 
		"hssv.args.octreeDepth", 
		"hssv.args.sceneScale", 
		"hssv.args.noCompression",
		"hssv.args.minNofEdgesNode");

	AABB const volume = createOctreeVolume();
	Octree* octree = vars.reCreate<Octree>("hssv.objects.octree", vars.getUint32("hssv.args.octreeDepth"), volume); 
  (void)octree;
	
	bool const forceBuild = vars.getBool("hssv.args.forceBuild");
	
	bool wasLoaded = false;
	if(!forceBuild)
	{
		std::cerr << "Trying to load octree from file...\n";

		wasLoaded = loadOctreeFromFile();

		if(!wasLoaded)
		{
			std::cerr << "Failed to load octree from file, building...\n";
		}
		else
		{
			std::cerr << "Octree succesfully loaded!\n";
		}
	}

	if(!wasLoaded)
	{
		buildOctree();

		if(!vars.getBool("hssv.args.dontStoreOctree"))
		{
			storeOctree();
		}
	}
}

void HSSV::buildOctree()
{
	Octree* octree = vars.get<Octree>("hssv.objects.octree");
	Adjacency* ad = vars.get<Adjacency>("adjacency");
	u32 const multiplicityBits = MathOps::getMaxNofSignedBits(vars.getUint32("maxMultiplicity"));
	u32 const minNofEdges = vars.getUint32("hssv.args.minNofEdgesNode");
	bool const isCompressed = !vars.getBool("hssv.args.noCompression");
	
	CpuBuilder builder;
	builder.fillOctree(octree, ad, multiplicityBits, isCompressed, minNofEdges);
}

bool HSSV::loadOctreeFromFile()
{
	Octree* octree = vars.get<Octree>("hssv.objects.octree");
	
	OctreeSerializer serializer;
	SerializerData data;
	data.modelName = vars.get<Model>("model")->getName();
	data.sceneScale = vars.getFloat("hssv.args.sceneScale");
	data.isCompressed = !vars.getBool("hssv.args.noCompression");
	data.deepestLevel = vars.getUint32("hssv.args.octreeDepth");
	data.minNofEdgesInSubnodes = vars.getUint32("hssv.args.minNofEdgesNode");

	return serializer.loadFromFile(octree, data);
}

void HSSV::storeOctree()
{
	Octree* octree = vars.get<Octree>("hssv.objects.octree");
	
	OctreeSerializer serializer;
	SerializerData data;
	data.modelName = vars.get<Model>("model")->getName();
	data.sceneScale = vars.getFloat("hssv.args.sceneScale");
	data.isCompressed = !vars.getBool("hssv.args.noCompression");
	data.deepestLevel = vars.getUint32("hssv.args.octreeDepth");
	data.minNofEdgesInSubnodes = vars.getUint32("hssv.args.minNofEdgesNode");

	serializer.storeToFile(octree, data);
}

void HSSV::createCapsDrawer()
{
	FUNCTION_PROLOGUE("hssv.objects", "renderModel");

	vars.reCreate<GSCaps>("hssv.objects.capsDrawer", vars);
}

void HSSV::resetMultiplicity()
{
	FUNCTION_PROLOGUE("hssv.objects", "renderModel", "maxMultiplicity");

	u32 const maxMult = vars.getUint32("maxMultiplicity");

	u32 const multiplicityBits = MathOps::getMaxNofSignedBits(maxMult);

	if(multiplicityBits > 8)
	{
		std::cerr << "Multiplicity too high! Resetting multiplicity to 2" << std::endl;
		*vars.get<u32>("maxMultiplicity") = 2;
		vars.updateTicks("maxMultiplicity");
	}
}

void HSSV::createSidesDrawer()
{
	FUNCTION_PROLOGUE("hssv.objects", "hssv.objects.octree", "hssv.args.drawCpu",);

	Octree* octree = vars.get<Octree>("hssv.objects.octree");
	Adjacency* ad = vars.get<Adjacency>("adjacency");
	u32 const maxMultiplicity = vars.getUint32("maxMultiplicity");

	if(vars.getBool("hssv.args.drawCpu"))
	{
		vars.reCreate<CpuSidesDrawer>("hssv.objects.sidesDrawer", octree, ad, maxMultiplicity);
	}
	else
	{
		vars.reCreate<GpuSidesDrawer>("hssv.objects.sidesDrawer", octree, ad, maxMultiplicity, vars);
	}
}

AABB HSSV::createOctreeVolume() const
{
	float const sceneScale = vars.getFloat("hssv.args.sceneScale");
	
	AABB bbox = getSceneAabb();
	fixVolume(bbox);

	glm::vec3 const extents = bbox.getDiagonal() * sceneScale;
	glm::vec3 center = bbox.getCenter();

	glm::vec3 minPoint = center - 0.5f * extents;
	glm::vec3 maxPoint = center + 0.5f * extents;

	bbox.setMin(minPoint);
	bbox.setMax(maxPoint);

	return bbox;
}

AABB HSSV::getSceneAabb() const
{
	std::vector<float> vertices = vars.get<Model>("model")->getVertices();
	glm::vec3* verts = reinterpret_cast<glm::vec3*>(&vertices[0]);

	size_t const nofVertices = vertices.size() / 3;

	AABB bbox;

	for (u32 i = 0; i < nofVertices; ++i)
	{
		bbox.addVertex(verts[i]);
	}

	return bbox;
}

void HSSV::fixVolume(AABB& bbox) const
{
	glm::vec3 e = bbox.getDiagonal();

	const float minSize = 0.01f;
	
	if (e.x < minSize)
	{
		glm::vec3 min = bbox.getMin();
		glm::vec3 max = bbox.getMax();

		bbox.setMin(min - glm::vec3(minSize, 0, 0));
		bbox.setMax(max + glm::vec3(minSize, 0, 0));
	}

	if (e.y < minSize)
	{
		glm::vec3 min = bbox.getMin();
		glm::vec3 max = bbox.getMax();

		bbox.setMin(min - glm::vec3(0, minSize, 0));
		bbox.setMax(max + glm::vec3(0, minSize, 0));
	}

	if (e.z < minSize)
	{
		glm::vec3 min = bbox.getMin();
		glm::vec3 max = bbox.getMax();

		bbox.setMin(min - glm::vec3(0, 0, minSize));
		bbox.setMax(max + glm::vec3(0, 0, minSize));
	}
}

void HSSV::buildTest()
{
	const std::vector<float> sceneScales = { 1, 2, 4, 8, 16 };
	const std::vector<uint32_t> octreeLevels = { 3, 4, 5 };

	std::ofstream saveFile;
	saveFile.open(std::string("buildTest_") + vars.get<Model>("model")->getName() + ".txt");

	for (const auto depth : octreeLevels)
	{
		for (const auto scale : sceneScales)
		{
			std::cerr << "\nDepth " << depth << " scale " << scale << std::endl;
			*(float*)(vars.get("hssv.args.sceneScale")) = scale;
			*(u32*)(vars.get("hssv.args.octreeDepth")) = depth;
			
			AABB const volume = createOctreeVolume();
			Octree* octree = vars.reCreate<Octree>("hssv.objects.octree", vars.getUint32("hssv.args.octreeDepth"), volume);

			HighResolutionTimer t;
			double buildTime = 0;
			t.reset();
			buildOctree();
			buildTime = t.getElapsedTimeSeconds();

			const size_t octreeSizeMB = octree->getOctreeSizeBytes() / 1024ull / 1024ull;

			std::string str;
			saveFile << std::scientific << depth << std::setw(15) << scale << std::setw(15) << buildTime << std::setw(10) << octreeSizeMB << "\n";
			std::cerr << str;
		}
	}

	std::cerr << "---END OF BUILD TEST---\n";

	saveFile.close();
}

