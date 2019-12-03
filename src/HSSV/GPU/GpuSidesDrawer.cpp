#include <GPU/GpuSidesDrawer.h>
#include <GPU/GpuShaderGenerator.h>
#include <Octree.h>
#include <AdjacencyWrapper.h>
#include <MathOps.h>

#include <FastAdjacency.h>
#include <FunctionPrologue.h>
#include <ifExistStamp.h>

#include <geGL/StaticCalls.h>

#include <fstream>
#include <algorithm>

using namespace ge::gl;

GpuSidesDrawer::GpuSidesDrawer(Octree* o, Adjacency* ad, u32 maxMultiplicity, vars::Vars& v) : SidesDrawerBase(o), vars(v)
{
	MaxMultiplicity = maxMultiplicity;
	Ad = ad;
	octree = o;
	NofBitsMultiplicity = MathOps::getMaxNofSignedBits(maxMultiplicity);

	CalcBitMasks8(2);
	CreateBuffers();

	std::cout << "GpuSidesDrawer consumes " << getGpuMemoryConsumptionMB() << "MB VRAM\n";
}

GpuSidesDrawer::~GpuSidesDrawer()
{
}

void GpuSidesDrawer::drawSides(const glm::mat4& mvp, const glm::vec4& light)
{
	s32 pos = GetLowestLevelCellPoint(glm::vec3(light));

	if(pos<0)
	{
		std::cerr << "Light out of range!\n";
		return;
	}

	CreateShaders();
	CreateEdgeRangeBuffer();

	ifExistStamp(vars, "");

	ComputeEdgeRanges(u32(pos));

	ifExistStamp(vars, "EdgeRanges");

	GenerateSidesFromRanges(light);

	ifExistStamp(vars, "SidesCompute");
	
	DrawSides(mvp);

	ifExistStamp(vars, "SidesDraw");
	//*/
}

u64 GpuSidesDrawer::getGpuMemoryConsumptionMB() const
{
	u64 sz = 0;

	sz += edgesBuffer->getSize();
	sz += oppositeVertices->getSize();
	sz += nofEdgesPrefixSumBuffer->getSize();
	sz += bitmaskBuffer->getSize();
	sz += DIBO[0]->getSize();
	sz += DIBO[1]->getSize();
	sz += edgeRangesBuffer->getSize();
	sz += IBO->getSize();
	sz += VBO->getSize();

	for(auto const& b : nodeEdgesIdBuffers)
	{
		sz += b->getSize();
	}

	return sz >> 20;
}

void GpuSidesDrawer::ComputeEdgeRanges(u32 lightNode)
{
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	DIBO[atomicIndex]->clear(GL_R32UI, 0, sizeof(u32), GL_RED, GL_INT);

	getEdgeRangesCs->use();
	getEdgeRangesCs->set1ui("nodeContainingLight", lightNode);
	getEdgeRangesCs->set1ui("nextKernelWgSize", vars.getUint32("hssv.args.wgSize"));

	nofEdgesPrefixSumBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	bitmaskBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	DIBO[atomicIndex]->bindRange(GL_SHADER_STORAGE_BUFFER, 2, 0, sizeof(u32));
	edgeRangesBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, 3);
	IBO->bindRange(GL_SHADER_STORAGE_BUFFER, 4, 0, sizeof(u32));

	glDispatchCompute(octree->getDeepestLevel() + 1, 1, 1);

	nofEdgesPrefixSumBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	bitmaskBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	DIBO[atomicIndex]->unbindRange(GL_SHADER_STORAGE_BUFFER, 2);
	edgeRangesBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, 3);
	IBO->unbindRange(GL_SHADER_STORAGE_BUFFER, 4);

	//--
	/*
	glFinish();

	std::vector<u32> v;
	edgeRangesBuffer->getData(v);

	std::vector<u32> ac;
	DIBO[atomicIndex]->getData(ac);

	u32 sumEdges = 0;
	for(u32 i = 1; i<2*ac[0]; i+=2)
	{
		sumEdges += (v[i] & 0x7FFFFFFF);
	}

	std::vector<u32> sh;
	shit->getData(sh);

	printf("");
	//*/

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GpuSidesDrawer::GenerateSidesFromRanges(glm::vec4 const& lightPosition)
{
	generateSidesCs->use();
	generateSidesCs->set4fv("lightPosition", glm::value_ptr(lightPosition));
	generateSidesCs->set1ui("nofEdges", u32(Ad->getNofEdges()));

	u32 bindSlot = 0;
	for(auto& buff : nodeEdgesIdBuffers)
	{
		buff->bindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	}

	edgesBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	oppositeVertices->bindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	edgeRangesBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	DIBO[atomicIndex]->bindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	IBO->bindRange(GL_SHADER_STORAGE_BUFFER, bindSlot++, 0, sizeof(u32));
	VBO->bindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	DIBO[atomicIndex^1]->bindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);

	DIBO[atomicIndex]->bind(GL_DISPATCH_INDIRECT_BUFFER);
	glDispatchComputeIndirect(0);

	DIBO[atomicIndex]->unbind(GL_DISPATCH_INDIRECT_BUFFER);

	bindSlot = 0;
	for (auto& buff : nodeEdgesIdBuffers)
	{
		buff->unbindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	}

	edgesBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	oppositeVertices->unbindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	edgeRangesBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	DIBO[atomicIndex]->unbindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	IBO->unbindRange(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	VBO->unbindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);
	DIBO[atomicIndex^1]->unbindBase(GL_SHADER_STORAGE_BUFFER, bindSlot++);

	atomicIndex ^= 1;
	//--
	/*
	glFinish();

	std::vector<u32> ib;
	IBO->getData(ib);

	std::vector<u32> sh;
	shit->getData(sh);

	printf("");
	//*/
	glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

void GpuSidesDrawer::DrawSides(glm::mat4 const& mvp)
{
	drawSidesProgram->use();
	drawSidesProgram->setMatrix4fv("mvp", glm::value_ptr(mvp));
	
	VAO->bind();
	IBO->bind(GL_DRAW_INDIRECT_BUFFER);

	glDrawArraysIndirect(GL_TRIANGLES, nullptr);

	VAO->unbind();
	IBO->unbind(GL_DRAW_INDIRECT_BUFFER);
}

void GpuSidesDrawer::CreateShaders()
{
	FUNCTION_PROLOGUE("hssv.objects", "hssv.args.wgSize", "hssv.objects.octree");

	CreateSidesDrawProgram();
	CreateEdgeRangeProgram();
	CreateSidesGenerationProgram();
}

void GpuSidesDrawer::CreateSidesDrawProgram()
{
	const char* vsSource = R"(
#version 430 core

layout(location=0)in vec4 Position;

uniform mat4 mvp = mat4(1);

void main(){
  gl_Position=mvp*Position;
}
)";
	drawSidesProgram = std::make_unique<Program>(std::make_shared<Shader>(GL_VERTEX_SHADER, vsSource));
}

void GpuSidesDrawer::CreateEdgeRangeProgram()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\getSubbuffersShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	//*/
	
	EdgeRangeShaderParams params;
	params.bitmaskBufferSize = GetNofIndicesPerBitmask();
	params.maxOctreeLevel = octree->getDeepestLevel();
	params.nofSubbuffers = TotalNofSubbuffers;
	params.subbufferCorrection = SubBufferCorrection;
	std::string const program = getComputeEdgeRangesCsSource(LastNodePerBuffer, params);
	//*/
	getEdgeRangesCs = std::make_unique<Program>(std::make_shared<Shader>(GL_COMPUTE_SHADER, program));
}

void GpuSidesDrawer::CreateSidesGenerationProgram()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\generateEdges.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	//*/
	
	SidesGenShaderParams params;
	params.bitmaskBufferSize = GetNofIndicesPerBitmask();
	params.maxOctreeLevel = octree->getDeepestLevel();
	params.nofBitsMultiplicity = NofBitsMultiplicity;
	params.wgSize = vars.getUint32("hssv.args.wgSize");
	std::string program = getComputeSidesFromEdgeRangesCsSource(LastNodePerBuffer, params);
	//*/
	generateSidesCs = std::make_unique<Program>(std::make_shared<Shader>(GL_COMPUTE_SHADER, program));
}

void GpuSidesDrawer::CreateBuffers()
{
	CreateEdgeBuffers();
	CreateLoadOctreeBuffers();
	CreateBitmaskBuffer();
	CreateDIBOs();
	CreateEdgeRangeBuffer();
	CreateDrawBuffers();
}

void GpuSidesDrawer::CreateEdgeBufferOLD()
{
	u32 const nofEdges = u32(Ad->getNofEdges());
	edgesBuffer = std::make_unique<Buffer>(nofEdges * (3 + 3 + 1 + MaxMultiplicity * 3) * sizeof(float));

	float * mappedBuf = reinterpret_cast<float*>(edgesBuffer->map(GL_WRITE_ONLY));
	u32 const edgeSize = 7 + 3 * MaxMultiplicity;

	for(u32 edge = 0; edge < nofEdges; ++edge)
	{
		glm::vec3 const& v1 = getEdgeVertexLow(Ad, edge);
		glm::vec3 const& v2 = getEdgeVertexHigh(Ad, edge);
		u32 const nofOpposite = getNofOppositeVertices(Ad, edge);

		mappedBuf[edge * edgeSize + 0] = v1.x;
		mappedBuf[edge * edgeSize + 1] = v1.y;
		mappedBuf[edge * edgeSize + 2] = v1.z;

		mappedBuf[edge * edgeSize + 3] = v2.x;
		mappedBuf[edge * edgeSize + 4] = v2.y;
		mappedBuf[edge * edgeSize + 5] = v2.z;

		reinterpret_cast<u32*>(mappedBuf)[edge * edgeSize + 6] = nofOpposite;

		for(u32 opposite = 0; opposite < nofOpposite; ++opposite)
		{
			glm::vec3 const& o = getOppositeVertex(Ad, edge, opposite);

			mappedBuf[edge * edgeSize + 7 + 3 * opposite + 0] = o.x;
			mappedBuf[edge * edgeSize + 7 + 3 * opposite + 1] = o.y;
			mappedBuf[edge * edgeSize + 7 + 3 * opposite + 2] = o.z;
		}
	}

	edgesBuffer->unmap();
}

void GpuSidesDrawer::CreateEdgeBuffers()
{
	u32 const nofEdges = u32(Ad->getNofEdges());
	edgesBuffer = std::make_unique<Buffer>(8u * nofEdges * sizeof(float));

	std::vector<float> oppositeVerts;
	oppositeVerts.reserve(3*nofEdges * MaxMultiplicity);

	float* mappedBuf = reinterpret_cast<float*>(edgesBuffer->map(GL_WRITE_ONLY));

	for (u32 edge = 0; edge < nofEdges; ++edge)
	{
		glm::vec3 const& v1 = getEdgeVertexLow(Ad, edge);
		glm::vec3 const& v2 = getEdgeVertexHigh(Ad, edge);
		u32 const nofOpposite = getNofOppositeVertices(Ad, edge);

		mappedBuf[edge * 3 + 0] = v1.x;
		mappedBuf[edge * 3 + 1] = v1.y;
		mappedBuf[edge * 3 + 2] = v1.z;

		mappedBuf[3*nofEdges + 3*edge + 0] = v2.x;
		mappedBuf[3*nofEdges + 3*edge + 1] = v2.y;
		mappedBuf[3*nofEdges + 3*edge + 2] = v2.z;

		reinterpret_cast<u32*>(mappedBuf)[6 * nofEdges + edge] = u32(oppositeVerts.size());
		reinterpret_cast<u32*>(mappedBuf)[7 * nofEdges + edge] = nofOpposite;

		for (u32 opposite = 0; opposite < nofOpposite; ++opposite)
		{
			glm::vec3 const& v = getOppositeVertex(Ad, edge, opposite);
			oppositeVerts.push_back(v.x);
			oppositeVerts.push_back(v.y);
			oppositeVerts.push_back(v.z);
		}
	}
	edgesBuffer->unmap();

	oppositeVertices = std::make_unique<Buffer>(oppositeVerts.size() * sizeof(float), oppositeVerts.data());
}

void GpuSidesDrawer::CreateLoadOctreeBuffers()
{
	u32 const nofNodes = octree->getTotalNumNodes();
	u64 const maxBufferSize = 2ull * 1024ull * 1024ull * 1024ull; //2GB

	std::vector<u32> nofEdgesPrefixSums;
	nofEdgesPrefixSums.reserve(2 * nofNodes * TotalNofSubbuffers + 1);

	u32 currentNode = 0;
	u64 remainingSize = octree->getOctreeSizeBytes();

	while (remainingSize > 0ull)
	{
		u64 const currentSize = remainingSize > maxBufferSize ? maxBufferSize : remainingSize;

		nofEdgesPrefixSums.push_back(0);

		std::shared_ptr<Buffer> buffer = std::make_shared<ge::gl::Buffer>(currentSize, nullptr);
		nodeEdgesIdBuffers.push_back(buffer);

		u32* dataPtr = reinterpret_cast<u32*>(glMapNamedBuffer(buffer->getId(), GL_WRITE_ONLY));

		u64 currentNumIndices = 0;
		while (currentNode < nofNodes)
		{
			Node* node = octree->getNode(currentNode);

			u64 const sz = octree->getNofAllEdgesInNode(currentNode);

			if ((sz + currentNumIndices) * sizeof(u32) > currentSize)
			{
				break;
			}

			//Pot edges
			for (u32 b = SubBufferCorrection; b <= BITMASK_ALL_SET; ++b)
			{
				u8 const bm = u8(b);
				auto it = node->edgesMayCastMap.find(bm);
				if (it != node->edgesMayCastMap.end() && !node->edgesMayCastMap[bm].empty())
				{
					u32 const subbufferSize = u32(node->edgesMayCastMap[bm].size());
					u32 const lastSum = nofEdgesPrefixSums[nofEdgesPrefixSums.size() - 1];
					memcpy(dataPtr + lastSum, node->edgesMayCastMap[bm].data(), subbufferSize * sizeof(u32));
					nofEdgesPrefixSums.push_back(lastSum + subbufferSize);
				}
				else
				{
					nofEdgesPrefixSums.push_back(nofEdgesPrefixSums[nofEdgesPrefixSums.size() - 1]);
				}
			}

			//Sil edges
			for (u32 b = SubBufferCorrection; b <= BITMASK_ALL_SET; ++b)
			{
				u8 bm = u8(b);
				auto it = node->edgesAlwaysCastMap.find(bm);
				if (it != node->edgesAlwaysCastMap.end() && !node->edgesAlwaysCastMap[bm].empty())
				{
					u32 const subbufferSize = u32(node->edgesAlwaysCastMap[bm].size());
					u32 const lastOffset = nofEdgesPrefixSums[nofEdgesPrefixSums.size() - 1];
					memcpy(dataPtr + lastOffset, node->edgesAlwaysCastMap[bm].data(), subbufferSize * sizeof(u32));
					nofEdgesPrefixSums.push_back(lastOffset + subbufferSize);
				}
				else
				{
					nofEdgesPrefixSums.push_back(nofEdgesPrefixSums[nofEdgesPrefixSums.size() - 1]);
				}
			}

			currentNumIndices += sz;
			currentNode++;
		}

		glUnmapNamedBuffer(buffer->getId());

		remainingSize -= currentNumIndices * sizeof(u32);
		
		LastNodePerBuffer.push_back(currentNode);
	}

	nofEdgesPrefixSumBuffer = std::make_unique<Buffer>(nofEdgesPrefixSums.size() * sizeof(u32), nofEdgesPrefixSums.data());
}

u32 GpuSidesDrawer::GetNofIndicesPerBitmask() const
{
	//m_bitMasks[0].size() - because all bitmask arrays are of same size
	return u32(BitmasksWithIthBitSet[0].size());
}

void GpuSidesDrawer::CreateBitmaskBuffer()
{
	u32 const nofIndices = GetNofIndicesPerBitmask();

	bitmaskBuffer = std::make_unique<Buffer>(8 * nofIndices * sizeof(u8));

	u8* buffer = reinterpret_cast<u8*>(bitmaskBuffer->map(GL_WRITE_ONLY));

	for(u8 i = 0; i < 8; ++i)
	{
		memcpy(buffer + i * nofIndices, BitmasksWithIthBitSet[i].data(), nofIndices * sizeof(u8));
	}

	bitmaskBuffer->unmap();
}

void GpuSidesDrawer::CreateDIBOs()
{
	typedef struct 
	{
		u32  num_groups_x;
		u32  num_groups_y;
		u32  num_groups_z;
	} DispatchIndirectCommand;

	DispatchIndirectCommand dic;
	dic.num_groups_x = 0;
	dic.num_groups_y = 1;
	dic.num_groups_z = 1;

	DIBO[0] = std::make_unique<Buffer>(sizeof(DispatchIndirectCommand), &dic);
	DIBO[1] = std::make_unique<Buffer>(sizeof(DispatchIndirectCommand), &dic);
}

void GpuSidesDrawer::CreateEdgeRangeBuffer()
{
	FUNCTION_PROLOGUE("hssv.objects", "hssv.args.wgSize", "hssv.objects.octree", "hssv.args.version");

	u32 const edgeRangeItems = nodeEdgesIdBuffers.size() > 1 ? 3 : 2;
	u32 const maxJobs = GetMaxNofJobs(vars.getUint32("hssv.args.wgSize"));

	if(edgeRangesBuffer)
	{
		edgeRangesBuffer.reset(nullptr);
	}

	edgeRangesBuffer = std::make_unique<Buffer>(2*maxJobs * edgeRangeItems * sizeof(u32));
}

void GpuSidesDrawer::CreateDrawBuffers()
{
	CreateIBO();
	CreateVBO();
	
	VAO = std::make_unique<VertexArray>();
	VAO->addAttrib(VBO, 0, 4, GL_FLOAT);
}

void GpuSidesDrawer::CreateIBO()
{
	DrawArraysIndirectCommand cmd;
	cmd.nofInstances = 1;

	IBO = std::make_unique<Buffer>(sizeof(DrawArraysIndirectCommand), &cmd);
}

void GpuSidesDrawer::CreateVBO()
{
	VBO = std::make_shared<Buffer>(Ad->getNofEdges() * MaxMultiplicity * 6 * 4 * sizeof(float));
}

void GpuSidesDrawer::CalcBitMasks8(unsigned int minBits)
{
	BitmasksWithIthBitSet.resize(OCTREE_NUM_CHILDREN);

	for (u32 i = 1; i < 256; ++i)
	{
		BitSet8 num = i;
		if (num.count() < minBits)
		{
			continue;
		}

		for (u32 b = 0; b < 8; ++b)
		{
			if (num[b])
			{
				BitmasksWithIthBitSet[b].push_back(u8(num.to_ulong()));
			}
		}
	}
}

u32 GpuSidesDrawer::GetMaxNodeNofJobsPotSil(u32 nodeID, u32 jobSize) const
{
	Node const* node = octree->getNode(nodeID);
	u32 ret = 0;

	if (node)
	{
		for(auto const& b : node->edgesAlwaysCastMap)
		{
			ret += u32(b.second.size()) / jobSize;
		}

		for (auto const& b : node->edgesMayCastMap)
		{
			ret += u32(b.second.size()) / jobSize;
		}
	}

	return ret;
}

u32 GpuSidesDrawer::GetMaxNofJobsInLevel(u32 level, u32 jobSize) const
{
	u32 const startingIndex = u32(octree->getLevelFirstNodeID(level));
	u32 const levelSize = octree->getLevelSize(level);

	u32 ret = 0;

	for (u32 i = 0; i < levelSize; ++i)
	{
		ret = std::max(ret, GetMaxNodeNofJobsPotSil(startingIndex + i, jobSize));
	}

	return ret;
}

u32 GpuSidesDrawer::GetMaxNofJobs(u32 jobSize) const
{
	u32 const deepestLevel = octree->getDeepestLevel();

	u32 ret = 0;

	for (u32 i = 0; i <= deepestLevel; ++i)
	{
		ret += GetMaxNofJobsInLevel(i, jobSize);
	}

	return ret;
}