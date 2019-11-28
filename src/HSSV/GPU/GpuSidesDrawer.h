#pragma once

#include <SidesDrawerBase.h>

#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/VertexArray.h>

class Adjacency;

class GpuSidesDrawer : public SidesDrawerBase
{
public:
	GpuSidesDrawer(Octree* octree, Adjacency* ad, u32 maxMultiplicity);
	~GpuSidesDrawer();

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
	void CreateEdgeBufferOLD();
	void CreateEdgeBuffers();
	void CreateBitmaskBuffer();
	void CreateAtomicCountersBuffer();
	void CreateEdgeRangeBuffer();
	void CreateDrawBuffers();
	void CreateIBO();
	void CreateVBO();

	void ClearAtomicCounterBuffer();
	void ClearIBO();

	u32 GetNofIndicesPerBitmask() const;
	void CalcBitMasks8(unsigned int minBits);

	void CalcSidesGenDispatchSize();
	glm::uvec2 GetMaxNofSubBuffersPotSil() const;
	glm::uvec2 GetNodeNofSubBuffersPotSil(uint32_t nodeID) const;
	glm::uvec2 GetMaxNofSubBuffersInLevelPotSil(uint32_t level) const;
	
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
	std::unique_ptr<ge::gl::Buffer> oppositeVertices;

	//Octree
	std::vector <std::shared_ptr<ge::gl::Buffer>> nodeEdgesIdBuffers;
	std::unique_ptr<ge::gl::Buffer> nofEdgesPrefixSumBuffer;

	//Data between programs
	std::unique_ptr<ge::gl::Buffer> bitmaskBuffer;
	std::unique_ptr<ge::gl::Buffer> atomicCounterBuffer[2];
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
	u32 NofWgsSidesGenDispatch = 0;
	u32 NofBitsMultiplicity;

	std::vector< std::vector<u8> > BitmasksWithIthBitSet;
	std::vector< u32 > LastNodePerBuffer;

	u32 atomicIndex = 0;
};