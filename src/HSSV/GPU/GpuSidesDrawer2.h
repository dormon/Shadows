#pragma once

#include <SidesDrawerBase.h>

#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/VertexArray.h>

#include<Vars/Vars.h>

class Adjacency;

class GpuSidesDrawer2 : public SidesDrawerBase
{
public:
	GpuSidesDrawer2(Octree* octree, Adjacency* ad, u32 maxMultiplicity, vars::Vars& vars);
	~GpuSidesDrawer2();

	void drawSides(const glm::mat4& mvp, const glm::vec4& light) override;

	u64 getGpuMemoryConsumptionMB() const;

private:
	void ComputeEdgeRanges(u32 lightNode);
	void GenerateSidesFromRanges(glm::vec4 const& lightPosition);
	void DrawSides(glm::mat4 const& mvp);

	void CreateShaders();
	void CreateSidesDrawProgram();
	void CreateEdgeRangeProgram();
	void CreateSidesGenerationProgram();

	void CreateBuffers();
	void CreateLoadOctreeBuffers();
	void CreateEdgeBuffer();
	void CreateBitmaskBuffer();
	void CreateDIBOs();
	void CreateEdgeRangeBuffer();
	void CreateDrawBuffers();
	void CreateIBO();
	void CreateVBO();

	u32 GetNofIndicesPerBitmask() const;
	void CalcBitMasks8(unsigned int minBits);

	u32 GetMaxNofJobs(u32 jobSize) const;
	u32 GetMaxNofJobsInLevel(uint32_t level, u32 jobSize) const;
	u32 GetMaxNodeNofJobsPotSil(uint32_t nodeID, u32 jobSize) const;
	
private:
	struct DrawArraysIndirectCommand
	{
		uint32_t nofVertices = 0;
		uint32_t nofInstances = 0;
		uint32_t firstVertex = 0;
		uint32_t baseInstance = 0;
	};

	std::unique_ptr<ge::gl::Program> getEdgeRangesCs;
	std::unique_ptr<ge::gl::Program> generateSidesCs;
	std::unique_ptr<ge::gl::Program> drawSidesProgram;

	//Edges
	std::unique_ptr<ge::gl::Buffer> edgesBuffer;

	//Octree
	std::vector <std::shared_ptr<ge::gl::Buffer>> nodeEdgesIdBuffers;
	std::unique_ptr<ge::gl::Buffer> nofEdgesPrefixSumBuffer;

	//Data between programs
	std::unique_ptr<ge::gl::Buffer> bitmaskBuffer;
	std::unique_ptr<ge::gl::Buffer> DIBO[2];
	std::unique_ptr<ge::gl::Buffer> edgeRangesBuffer;

	//Drawing stuff
	std::unique_ptr<ge::gl::Buffer>      IBO;
	std::shared_ptr<ge::gl::Buffer>      VBO;
	std::unique_ptr<ge::gl::VertexArray> VAO;

	Adjacency* Ad;
	Octree* octree;
	u32 MaxMultiplicity;

	//skipping first 3 1-bit numbers
	u32 TotalNofSubbuffers = 253; 
	u32 SubBufferCorrection = 3; 
	u32 NofBitsMultiplicity;

	std::vector< std::vector<u8> > BitmasksWithIthBitSet;
	std::vector< u32 > LastNodePerBuffer;

	u32 atomicIndex = 0;

	vars::Vars& vars;
};