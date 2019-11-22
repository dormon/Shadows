#include <HSSV.h>
#include <Octree.h>
#include <CPU/CpuBuilder.h>

#include <GSCaps.h>
#include <Model.h>
#include <FunctionPrologue.h>
#include <createAdjacency.h>

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
	buildOctree();
}

void HSSV::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	createCapsDrawer();

	vars.get<GSCaps>("hssv.objects.capsDrawer")->drawCaps(lightPosition, viewMatrix, projectionMatrix);
}

void HSSV::drawUser(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{

}

void HSSV::buildOctree()
{
	FUNCTION_PROLOGUE("hssv.objects", "renderModel", "hssv.args.buildOnCpu", "hssv.args.octreeDepth", "hssv.args.sceneScale", "hssv.args.compressionLevel");

	AABB const volume = createOctreeVolume();
	Octree* octree = vars.reCreate<Octree>("hssv.objects.octree", vars.getUint32("hssv.args.octreeDepth"), volume);

	if(vars.getBool("hssv.args.buildOnCpu"))
	{
		CpuBuilder builder;
		builder.fillOctree(octree, vars.get<Adjacency>("adjacency"));
	}
	else
	{

	}
}

void HSSV::createCapsDrawer()
{
	FUNCTION_PROLOGUE("hssv.objects", "renderModel");

	vars.reCreate<GSCaps>("hssv.objects.capsDrawer", vars);
}

//We want max multiplicity of 2 at all times
void HSSV::resetMultiplicity()
{
	//PREROBIT na 24-bit
	FUNCTION_PROLOGUE("hssv.objects", "renderModel", "maxMultiplicity");

	uint32_t const maxMult = vars.getUint32("maxMultiplicity");

	if(maxMult!=2)
	{
		*vars.get<uint32_t>("maxMultiplicity") = 2;
		vars.updateTicks("maxMultiplicity");
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

	for (unsigned int i = 0; i < nofVertices; ++i)
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

