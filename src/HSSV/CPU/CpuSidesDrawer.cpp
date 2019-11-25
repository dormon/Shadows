#include <CPU/CpuSidesDrawer.h>
#include <Octree.h>
#include <FastAdjacency.h>
#include <MultiplicityCoder.h>
#include <AdjacencyWrapper.h>
#include <MathOps.h>

#include <geGL/StaticCalls.h>

#include <algorithm>
#include <iostream>
#include <fstream>

using namespace ge::gl;

constexpr u32 NOF_VERTEX_COMPONENTS = 4u;

const char* vsSourceHssv = R"(
#version 430 core

layout(location=0)in vec4 Position;

uniform mat4 mvp = mat4(1);

void main()
{
	gl_Position=mvp*Position;
}
)";

CpuSidesDrawer::CpuSidesDrawer(Octree* o, Adjacency* ad, u32 maxMultiplicity) : SidesDrawerBase(o)
{
	SidesProgram = std::make_unique<Program>(std::make_shared<Shader>(GL_VERTEX_SHADER, vsSourceHssv));

	Ad = ad;
	PrepareBuffers(maxMultiplicity * Ad->getNofEdges() * 6 * NOF_VERTEX_COMPONENTS * sizeof(float));

	NofBitsMultiplicity = MathOps::getMaxNofSignedBits(maxMultiplicity);
}


CpuSidesDrawer::~CpuSidesDrawer()
{
}

void CpuSidesDrawer::PrepareBuffers(size_t maxVboSize)
{
	VBO = std::make_shared <ge::gl::Buffer>(maxVboSize);
	VAO = std::make_shared<ge::gl::VertexArray>();

	VAO->addAttrib(VBO, 0, 4, GL_FLOAT);
}


void CpuSidesDrawer::drawSides(const glm::mat4& mvp, const glm::vec4& light)
{
	std::vector<float> sidesGeometry = GetSilhouetteFromLightPos(light);
	
	UpdateSidesVBO(sidesGeometry);
	
	VAO->bind();
	SidesProgram->use();
	SidesProgram->setMatrix4fv("mvp", glm::value_ptr(mvp));
	
	ge::gl::glDrawArrays(GL_TRIANGLES, 0, GLsizei(sidesGeometry.size() / NOF_VERTEX_COMPONENTS));
	
	VAO->unbind();
}

void CpuSidesDrawer::UpdateSidesVBO(const std::vector<float>& vertices)
{
	float* ptr = reinterpret_cast<float*>(VBO->map(GL_MAP_WRITE_BIT));

	memcpy(ptr, vertices.data(), vertices.size() * sizeof(float));

	VBO->unmap();
}


CpuSidesDrawer::Edges CpuSidesDrawer::GetSilhouttePotentialEdgesFromNodeUp(uint32_t nodeID)
{
	u32 currentNodeID = nodeID;
	s32 currentLevel = octree->getDeepestLevel();
	u8 cameAsChildId = 0;

	Edges edges;

	if (printTraversePath) std::cerr << "Traversal stats: \n";

	while (currentLevel >= 0)
	{
		Node const*  node = octree->getNode(currentNodeID);

		assert(node != nullptr);

		if (printTraversePath) std::cerr << "---Node " << currentNodeID << ":\n";

		if (printTraversePath) std::cerr << "Sil nodes:\n";
		for(auto const& edgeBuffer : node->edgesAlwaysCastMap)
		{
			if((edgeBuffer.first >> cameAsChildId) & 1)
			{
				edges.silhouette.insert(edges.silhouette.end(), edgeBuffer.second.begin(), edgeBuffer.second.end());

				if (printTraversePath) std::cerr << "Mask " << u32(edgeBuffer.first) << ": " << edgeBuffer.second.size() << " edges\n";
			}
		}

		if (printTraversePath) std::cerr << "Pot nodes:\n";
		for (auto const& edgeBuffer : node->edgesMayCastMap)
		{
			if ((edgeBuffer.first >> cameAsChildId) & 1)
			{
				edges.potential.insert(edges.potential.end(), edgeBuffer.second.begin(), edgeBuffer.second.end());

				if (printTraversePath) std::cerr << "Mask " << u32(edgeBuffer.first) << ": " << edgeBuffer.second.size() << " edges\n";
			}
		}

		--currentLevel;
		if(currentLevel>=0)
		{
			cameAsChildId = u8(octree->getNodeIndexWithinParent(currentNodeID));
			currentNodeID = octree->getNodeParent(currentNodeID);
		}
	}

	printTraversePath = false;
	return std::move(edges);
}

std::vector<float> CpuSidesDrawer::GetSilhouetteFromLightPos(const glm::vec3& lightPos)
{
	std::vector<float> sidesVertices;
	
	s32 const lowestNode = GetLowestLevelCellPoint(lightPos);

	if (lowestNode < 0)
	{
		AABB vol = octree->getNodeVolume(0);
		std::cerr << "Light (" << lightPos.x << " " << lightPos.y << " " << lightPos.z << ") is out of octree range min:" << vol.getMin().x << " " << vol.getMin().y << " " << vol.getMin().z 
			<<" max : " << vol.getMax().x << " " << vol.getMax().y << " " << vol.getMax().z<< std::endl;
		return std::vector<float>();
	}

	std::vector<uint32_t> castPot;

	Edges e = GetSilhouttePotentialEdgesFromNodeUp(u32(lowestNode));

	sidesVertices.reserve((e.potential.size() + e.silhouette.size()) * NOF_VERTEX_COMPONENTS);

	MultiplicityCoder mc(NofBitsMultiplicity);

	for (const auto edge : e.silhouette)
	{
		u32 const edgeId = mc.decodeEdgeFromEncoded(edge);
		
		s32 const multiplicity = MathOps::calcEdgeMultiplicity(Ad, edgeId, lightPos);
		//s32 const multiplicity = mc.decodeEdgeMultiplicityFromId(edge);

		glm::vec3 const& lowerPoint = getEdgeVertexLow(Ad, edgeId);
		glm::vec3 const& higherPoint = getEdgeVertexHigh(Ad, edgeId);

		GeneratePushSideFromEdge(lightPos, lowerPoint, higherPoint, multiplicity, sidesVertices);
	}
	
	for (const auto edge : e.potential)
	{
		const int multiplicity = MathOps::calcEdgeMultiplicity(Ad, edge, lightPos);
		if (multiplicity != 0)
		{
			const glm::vec3& lowerPoint = getEdgeVertexLow(Ad, edge);
			const glm::vec3& higherPoint = getEdgeVertexHigh(Ad, edge);

			GeneratePushSideFromEdge(lightPos, lowerPoint, higherPoint, multiplicity, sidesVertices);
			castPot.push_back(mc.encodeEdgeMultiplicityToId(edge, multiplicity));
		}
	}
	
	if (printEdgeStats)
	{
		//--
		std::ofstream vstream;
		vstream.open("CPuVerts.txt");

		for(unsigned int i=0; i<sidesVertices.size(); i+=4)
		{
			vstream << sidesVertices[i] << " " << sidesVertices[i+1] << " " << sidesVertices[i+2] << " " << sidesVertices[i+3] << std::endl;
		}
		vstream.close();
		//--
		
		
		std::cout << "Light node: " << lowestNode << std::endl;
		std::cout << "Light " << lightPos.x << ", " << lightPos.y << ", " << lightPos.z << std::endl;
		const auto n = octree->getNode(lowestNode);
		auto minP = n->volume.getMin();
		auto maxP = n->volume.getMax();
		std::cout << "Node space " << minP.x << ", " << minP.y << ", " << minP.z << " Max: " << maxP.x << ", " << maxP.y << ", " << maxP.z << "\n";
		minP = n->volume.getCenter();
		maxP = n->volume.getDiagonal();
		std::cout << "Center " << minP.x << ", " << minP.y << ", " << minP.z << " Extents: " << maxP.x << ", " << maxP.y << ", " << maxP.z << "\n";

		std::cout << "Num potential: " << e.potential.size() << ", numSilhouette: " << e.silhouette.size() << std::endl;
		std::cout << "Silhouette consists of " << e.silhouette.size() + castPot.size() << " edges\n";

		std::ofstream of;
		of.open("CPU_Edges.txt");
		std::sort(e.potential.begin(), e.potential.end());
		std::sort(e.silhouette.begin(), e.silhouette.end());
		std::sort(castPot.begin(), castPot.end());

		of << "Potential:\n";
		for (const auto e : e.potential)
			of << e << std::endl;

		of << "\nSilhouette:\n";
		for (const auto e : e.silhouette)
			of << mc.decodeEdgeFromEncoded(e) << " multiplicity: " << mc.decodeEdgeMultiplicityFromId(e) << std::endl;

		of << "\nUsedPot:\n";
		for (auto const cp : castPot)
			of << mc.decodeEdgeFromEncoded(cp) << " multiplicity: " << mc.decodeEdgeMultiplicityFromId(cp) << std::endl;

		of.close();
		
		//std::ofstream sof;
		//sof.open("silhouette.txt");
		//sof << "SIL\n";
		//for (const auto e : silhouetteEdges)
		//sof << decodeEdgeFromEncoded(e) << "(" << decodeEdgeMultiplicityFromId(e) << ")" << std::endl;
		//sof << "POT\n";
		//for (const auto e : ed)
		//sof << decodeEdgeFromEncoded(e) << "(" << decodeEdgeMultiplicityFromId(e) << ")" << std::endl;
		//sof.close();
	

		printEdgeStats = false;
	}
	//*/

	return sidesVertices;
}

void CpuSidesDrawer::GeneratePushSideFromEdge(const glm::vec3& lightPos, const glm::vec3& lowerPoint, const glm::vec3& higherPoint, int multiplicity, std::vector<float>& sides)
{
	const glm::vec3 lowInfinity = lowerPoint - lightPos;
	const glm::vec3 highInfinity = higherPoint - lightPos;

	const uint32_t absMultiplicity = abs(multiplicity);

	if (multiplicity > 0)
	{
		for (uint32_t i = 0; i < absMultiplicity; ++i)
		{
			sides.push_back(highInfinity.x); sides.push_back(highInfinity.y); sides.push_back(highInfinity.z); sides.push_back(0);
			sides.push_back(higherPoint.x); sides.push_back(higherPoint.y); sides.push_back(higherPoint.z); sides.push_back(1.0f);
			sides.push_back(lowerPoint.x); sides.push_back(lowerPoint.y); sides.push_back(lowerPoint.z); sides.push_back(1.0f);

			sides.push_back(lowInfinity.x); sides.push_back(lowInfinity.y); sides.push_back(lowInfinity.z); sides.push_back(0);
			sides.push_back(highInfinity.x); sides.push_back(highInfinity.y); sides.push_back(highInfinity.z); sides.push_back(0);
			sides.push_back(lowerPoint.x); sides.push_back(lowerPoint.y); sides.push_back(lowerPoint.z); sides.push_back(1.0f);
		}
	}
	else if (multiplicity < 0)
	{
		for (uint32_t i = 0; i < absMultiplicity; ++i)
		{
			sides.push_back(lowInfinity.x); sides.push_back(lowInfinity.y); sides.push_back(lowInfinity.z); sides.push_back(0);
			sides.push_back(lowerPoint.x); sides.push_back(lowerPoint.y); sides.push_back(lowerPoint.z); sides.push_back(1.0f);
			sides.push_back(higherPoint.x); sides.push_back(higherPoint.y); sides.push_back(higherPoint.z); sides.push_back(1.0f);

			sides.push_back(highInfinity.x); sides.push_back(highInfinity.y); sides.push_back(highInfinity.z); sides.push_back(0);
			sides.push_back(lowInfinity.x); sides.push_back(lowInfinity.y); sides.push_back(lowInfinity.z); sides.push_back(0);
			sides.push_back(higherPoint.x); sides.push_back(higherPoint.y); sides.push_back(higherPoint.z); sides.push_back(1.0f);
		}
	}
}