#pragma once

#include <SidesDrawerBase.h>

#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/VertexArray.h>

class Adjacency;

class CpuSidesDrawer : public SidesDrawerBase
{
public:

	CpuSidesDrawer(Octree* octree, Adjacency* ad, u32 maxMultiplicity);
	virtual ~CpuSidesDrawer();

	void drawSides(const glm::mat4& mvp, const glm::vec4& light) override;

private:

	std::vector<float> GetSilhouetteFromLightPos(const glm::vec4& lightPos);
	void GeneratePushSideFromEdge(const glm::vec4& lightPos, const glm::vec3& lowerPoint, const glm::vec3& higherPoint, int multiplicitySign, std::vector<float>& sides);

	void PrepareBuffers(size_t maxVboSize);
	void UpdateSidesVBO(const std::vector<float>& vertices);

	struct Edges
	{
		std::vector<uint32_t> potential;
		std::vector<uint32_t> silhouette;
	};

	Edges GetSilhouttePotentialEdgesFromNodeUp(uint32_t nodeID);

private:
	std::shared_ptr<ge::gl::Buffer> VBO;
	std::shared_ptr<ge::gl::VertexArray> VAO;
	std::unique_ptr<ge::gl::Program> SidesProgram;

	Adjacency* Ad;
	u32 NofBitsMultiplicity;

	bool printTraversePath = true;
	bool printEdgeStats = true;
};
