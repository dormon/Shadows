#include <GPU/GpuShaderGenerator.h>

#include <sstream>

std::string getComputeEdgeRangesCsSource(std::vector<u32> const& lastNodePerBuffer, EdgeRangeShaderParams const& info)
{
	std::stringstream str;

	bool const useExtendedVersion = lastNodePerBuffer.size() > 1;
	u32 const numBuffers = u32(lastNodePerBuffer.size());
  (void)numBuffers;

	str << "#version 450 core\n\n";
	str << "#define MAX_OCTREE_LEVEL " << info.maxOctreeLevel << "u\n";
	str << "#define EDGERANGE_INFO_SIZE " << (useExtendedVersion ? 3 : 2) << "u\n";
	str << "#define NOF_SUBBUFFERS " << info.nofSubbuffers << "u\n";
	str << "#define SUBBUFF_CORRECTION " << info.subbufferCorrection << "u\n";
	str << "#define BITMASK_BUFFER_SIZE " << info.bitmaskBufferSize << "u\n";

	str << R".(
#define OCTREE_NUM_CHILDREN 8u
#define WG_SIZE 256

#define POT_INDEX 0u
#define SIL_INDEX 1u

layout (local_size_x = WG_SIZE) in;

layout(std430, binding = 0) restrict readonly  buffer _nofEdgesPrefixSum{ uint nofEdgesPrefixSum[]; };
layout(std430, binding = 1) restrict readonly  buffer _bitmasks         { uint bitmasks[254]; }; //stored as bytes, 8*127 / 4
layout(std430, binding = 2) restrict           buffer _atomicCounters   { uint globalAtomicCounter; };
layout(std430, binding = 3) restrict writeonly buffer _edgeRangeInfo    { uint edgeRangeInfo[]; }; //2*MAX_OCTREE_LEVEL*BITMASK_BUFFER_SIZE * EDGERANGE_INFO_SIZE
layout(std430, binding = 4) restrict writeonly buffer _clr    { uint clr; }; 

uint getNodeParent(uint nodeID)
{	
	return uint(floor((nodeID - 1.f) / OCTREE_NUM_CHILDREN));
}

uint getChildrenStartingId(uint nodeID)
{
	return OCTREE_NUM_CHILDREN * nodeID + 1;
}

uint getNodeIndexWithinParent(uint nodeID, uint parent)
{
	const uint startID = getChildrenStartingId(parent);

	return nodeID - startID;
}

uvec2 getNodeBitmaskEdgeStartCount(uint nodeId, uint mask, uint isSil, uint offset)
{
	const uint index = (2 * NOF_SUBBUFFERS * nodeId) + (isSil * NOF_SUBBUFFERS) + (mask - SUBBUFF_CORRECTION) + offset;
	const uint start = nofEdgesPrefixSum[index];
	const uint end = nofEdgesPrefixSum[index+1];
	
	return uvec2(start, end - start);
}
//------------------------------------------------

uniform uint nodeContainingLight;
uniform uint maxJobSize;

shared uint localAtomicCounter;
shared uint globalOffset;

).";

	if(useExtendedVersion)
	{
		str << R".(
void storeSubbufferEdgeRange(uint start, uint edgeCount, uint index, uint bufferNum)
{
	const uint storePos = index * EDGERANGE_INFO_SIZE;
	edgeRangeInfo[storePos + 0] = start;
	edgeRangeInfo[storePos + 1] = edgeCount;
	edgeRangeInfo[storePos + 2] = bufferNum;
}
).";

	}
	else
	{
		str << R".(
void storeSubbufferEdgeRange(uint start, uint edgeCount, uint index)
{
	const uint storePos = index * EDGERANGE_INFO_SIZE;
	edgeRangeInfo[storePos + 0] = start;
	edgeRangeInfo[storePos + 1] = edgeCount;
}
).";
	}

	if (useExtendedVersion)
	{
		str << genEdgeBuffersMappingFn(lastNodePerBuffer);
	}

	str << R".(
uint encodeNofEdgesIsSil(uint nofEdges, uint isSil)
{
	return nofEdges | (isSil * 0x80000000 ); //set the highest bit
}

void main()
{
	//Select buffers
	const uint wgId = gl_WorkGroupID.x;
	const uint localId = gl_LocalInvocationID.x;
	
	if(wgId > MAX_OCTREE_LEVEL)
	{
		return;
	}
	
	if(localId==0)
	{
		localAtomicCounter = 0;
		clr = 0;
	}

	barrier();
	
	uint childBit = 0;
	uint myNode = nodeContainingLight;
	
	//Get node to process and child bit
	for(uint i = 0; i< wgId; ++i)
	{
		uint parent = getNodeParent(myNode);
		childBit = getNodeIndexWithinParent(myNode, parent);
		myNode = parent;
	}

	//Grab an bitmask and test if empty
	uint storeIndex = 0;
	uint nofEdges = 0;
	uint edgesStart = 0;
	uint nofJobs = 0;
	const uint isSil = (localId<BITMASK_BUFFER_SIZE) ? POT_INDEX : SIL_INDEX;
).";

	if(useExtendedVersion)
	{
		str << "	uint bufferNum = getBufferNum(myNode);\n";
	}

	str << R".(
	if(localId < 2*BITMASK_BUFFER_SIZE)
	{
		const uint pos = localId % BITMASK_BUFFER_SIZE;
		const uint bigIndex = childBit * BITMASK_BUFFER_SIZE + pos;
		const uint index = bigIndex >> 2; // div 4
		const uint shift = bigIndex &  0x3; // mod 4
		const uint mask = (bitmasks[index] >> 8*shift) & 0x000000FF;
	
		//Test bitmask
).";
		if(useExtendedVersion)
		{
			str << "		uvec2 startAndNofEdges = getNodeBitmaskEdgeStartCount(myNode, mask, isSil, bufferNum);\n";
		}
		else
		{
			str << "		uvec2 startAndNofEdges = getNodeBitmaskEdgeStartCount(myNode, mask, isSil, 0);\n";
		}
		
	str << R".(	
		nofEdges = startAndNofEdges.y;
		edgesStart = startAndNofEdges.x;	 
		if(nofEdges > 0)
		{
			nofJobs = (nofEdges / maxJobSize) + uint((nofEdges % maxJobSize) != 0);
			storeIndex = atomicAdd(localAtomicCounter, nofJobs);
		}
	}

	barrier();
	
	if(localId==0)
	{
		globalOffset = atomicAdd(globalAtomicCounter, localAtomicCounter);
	}
	
	barrier();
	
	if(nofEdges > 0 )
	{
		storeIndex += globalOffset;
		uint remainigEdges = nofEdges;
		for(uint i = 0; i< nofJobs; ++i)
		{
			const uint storeSize = (remainigEdges > maxJobSize) ? maxJobSize : remainigEdges;
).";
	if (useExtendedVersion)
	{
		str << "		storeSubbufferEdgeRange(edgesStart + i * maxJobSize, encodeNofEdgesIsSil(storeSize, isSil), storeIndex +i, bufferNum);\n"; 
	}
	else
	{
		str << "		storeSubbufferEdgeRange(edgesStart + i * maxJobSize, encodeNofEdgesIsSil(storeSize, isSil), storeIndex + i);\n";
	}

	str << R".(
			remainigEdges -= storeSize;
		}
	}
}
).";

	return str.str();
}

std::string genEdgeBuffersMappingFn(std::vector<u32> const& lastNodePerBuffer)
{
	std::stringstream str;
	u32 const nofEdgeBuffers = u32(lastNodePerBuffer.size());

	str << "const uint edgeBuffersMapping[" << nofEdgeBuffers << "] = uint[" << nofEdgeBuffers << "](";
	for (u32 i = 0; i < nofEdgeBuffers; ++i)
	{
		str << lastNodePerBuffer[i];
		if (i != (nofEdgeBuffers - 1))
			str << ", ";
	}
	str << ");\n";

	str << "uint getBufferNum(uint edgeId)\n{\n";
	for (u32 i = 0; i < nofEdgeBuffers; ++i)
	{
		str << "	if(edgeId <= edgeBuffersMapping[" << i << "]) return " << i << ";\n";
	}
	str << "	return 0;\n}\n";

	return str.str();
}


std::string getComputeSidesFromEdgeRangesCsSource(std::vector<u32> const& lastNodePerBuffer, SidesGenShaderParams const& params)
{
	std::stringstream str;

	bool const useExtendedVersion = lastNodePerBuffer.size() > 1;
	u32 const nofEdgeBuffers = u32(lastNodePerBuffer.size());
	
	str << R".(
#version 450 core

#extension GL_ARB_shader_ballot : enable
#extension GL_KHR_shader_subgroup_basic : enable

).";

	str << "#define EDGERANGE_INFO_SIZE " << (useExtendedVersion ? 3 : 2) << "u\n";
	str << "#define BITMASK_BUFFER_SIZE " << params.bitmaskBufferSize << "u\n";
	str << "#define MULTIPLICITY_BITS " << params.nofBitsMultiplicity << "\n";
	str << "#define MAX_MULTIPLICITY " << params.maxMultiplicity << "\n";
	str << "#define WG_SIZE " << params.wgSize << "u\n";
	str << "#define NOF_VEC4_EDGE " << params.edgeSizeNofVec4 << "u\n";

	str << R".(
#define NOF_DATA_PER_SIDE 1
#define MULTIPLICITY_MASK ((uint(1) << MULTIPLICITY_BITS) - 1)
#define POT_INDEX 0u
#define SIL_INDEX 1u

).";

	u32 bindSlot = 0;
	for (; bindSlot < nofEdgeBuffers; ++bindSlot)
	{
		str << "layout(std430, binding = " << bindSlot << ") restrict readonly  buffer _edges" << bindSlot << " { uint edgeIndices" << bindSlot << "[]; };\n";
	}

	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _edgesInfos { vec4 edges[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _edgeRanges { uint edgeRanges[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _nofPotSilBuffers { uint nofPotSilJobs; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _indirectData { uint nofIndicesToDraw; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _computedMults { uint mults[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _clr { uint clr; };\n";

	str << R".(
layout (local_size_x = WG_SIZE) in;

uniform vec4 lightPosition;

).";
	if (useExtendedVersion)
	{
		str << R".(
uvec3 getEdgeRange(uint index)
{
	const uint pos = index * EDGERANGE_INFO_SIZE;
	return uvec3(edgeRanges[pos + 0], edgeRanges[pos + 1], edgeRanges[pos + 2]);
}
).";
	}
	else
	{
		str << R".(
uvec3 getEdgeRange(uint index)
{
	const uint pos = index * EDGERANGE_INFO_SIZE;
	return uvec3(edgeRanges[pos + 0], edgeRanges[pos + 1], 0);
}
).";
	}

	str << R".(
uint encodeEdgeMultiplicityToId(uint edgeID, int multiplicity)
{
	return uint(edgeID << MULTIPLICITY_BITS) | (multiplicity & MULTIPLICITY_MASK);
}
).";

	if (useExtendedVersion)
	{
		str << R".(
uint getEdgeId(uint pos, uint buf)
{	
).";
		for (u32 i = 0; i < nofEdgeBuffers; ++i)
		{
			str << "	if(buf== " << i << ") {return edgeIndices" << i << "[pos];}\n";
		}
		str << "	return 0;\n}\n";
	}
	else
	{
		str << R".(
uint getEdgeId(uint pos, uint unused)
{
	return edgeIndices0[pos];
}
).";
	}

	str << R".(

void pushEdge(uint storeIndex, uint encodedEdgeId)
{
	mults[storeIndex] = encodedEdgeId;
}

shared uint warpCounters[32]; //max 32 warps per WG on nV - 1024 threads

uint decodeWorkSize(uint val)
{
	return val & 0x7FFFFFFF;
}

uint decodeIsSil(uint val)
{
	return uint((val & 0x80000000) !=0);
}

uint getPotEdgeStorePos(uint dataPerSide, uint warpId)
{
	if(gl_SubGroupInvocationARB == 0)
	{
		warpCounters[warpId] = 0;
	}
	
	const uint localOffset = atomicAdd(warpCounters[warpId], dataPerSide);
	
	uint globalOffset = 0;
	if(gl_SubGroupInvocationARB == 0)
	{
		globalOffset = atomicAdd(nofIndicesToDraw, warpCounters[warpId]);
	}
	
	globalOffset = readFirstInvocationARB(globalOffset);
	
	return globalOffset + localOffset;
}

uint getSilEdgeStorePos(uint localId, uint jobSize)
{
	if(localId == 0)
	{
		warpCounters[0] = atomicAdd(nofIndicesToDraw, jobSize);
	}

	barrier();

	return localId + warpCounters[0];
}

void main()
{
	const uint wgId = gl_WorkGroupID.x;
	const uint localId = gl_LocalInvocationID.x;
	const uint warpId = localId / gl_SubGroupSizeARB;
	
	if(wgId >= nofPotSilJobs)
	{
		return;
	}
	
	if(gl_GlobalInvocationID.x == 0)
	{
		clr = 0;
	}

	const uvec3 job = getEdgeRange(wgId);
	const uint jobSize = decodeWorkSize(job.y);
	const uint jobStart = job.x;
	const uint isSil = decodeIsSil(job.y);
	
	if(localId >= jobSize)
	{
		return;
	}
	
	if(isSil==SIL_INDEX)
	{
		const uint edgeId = getEdgeId(jobStart + localId, job.z);
		uint storeIndex = getSilEdgeStorePos(localId, jobSize);
		
		pushEdge(storeIndex, edgeId);
	}
	else
	{
		const uint edgeId = getEdgeId(jobStart + localId, job.z);
		const uint edgeStart = edgeId * NOF_VEC4_EDGE;
		
		vec4 oppositePlanes[MAX_MULTIPLICITY];
		vec4 v[2];
		
		v[0] = edges[edgeStart + 0]; //edgeLow.xyz, empty
		v[1] = edges[edgeStart + 1]; //edgeHigh.xyz, nofOpposite
		
		for(uint o = 0; o < MAX_MULTIPLICITY; ++o)
		{
			oppositePlanes[o] = edges[edgeStart + 2 + o];
		}
		
		int multiplicity = 0;
		
		for(uint ov = 0; ov < MAX_MULTIPLICITY; ++ov)
		{
			multiplicity += int(sign(dot(oppositePlanes[ov],lightPosition)));
		}
		
		const uint storeIndex = getPotEdgeStorePos(uint(multiplicity!=0)*NOF_DATA_PER_SIDE, warpId);
		
		if(multiplicity!=0)
		{
			pushEdge(storeIndex, encodeEdgeMultiplicityToId(edgeId, multiplicity));
		}
	}
}
).";

	return str.str();
}

std::string getComputeSidesFromEdgeRangesCsSourceAdvanced(std::vector<u32> const& lastNodePerBuffer, SidesGenShaderParams const& params)
{
	std::stringstream str;

	bool const useExtendedVersion = lastNodePerBuffer.size() > 1;
	u32 const nofEdgeBuffers = u32(lastNodePerBuffer.size());

	str << R".(
#version 450 core

#extension GL_ARB_shader_ballot : enable
#extension GL_KHR_shader_subgroup_basic : enable

).";

	str << "#define EDGERANGE_INFO_SIZE " << (useExtendedVersion ? 3 : 2) << "u\n";
	str << "#define BITMASK_BUFFER_SIZE " << params.bitmaskBufferSize << "u\n";
	str << "#define MULTIPLICITY_BITS " << params.nofBitsMultiplicity << "\n";
	str << "#define MAX_MULTIPLICITY " << params.maxMultiplicity << "\n";
	str << "#define WG_SIZE " << params.wgSize << "u\n";
	str << "#define NOF_VEC4_EDGE " << params.edgeSizeNofVec4 << "u\n";

	str << R".(
#define NOF_DATA_PER_SIDE 1
#define MULTIPLICITY_MASK ((uint(1) << MULTIPLICITY_BITS) - 1)

#define BIG_POT_ITERS ((2+MAX_MULTIPLICITY) / 4)
#define BIG_LEFTOVER ((2+MAX_MULTIPLICITY) % 4)
#define SMALL_POT_ITERS ((BIG_LEFTOVER/2) + (BIG_LEFTOVER & 1))

#define POT_INDEX 0u
#define SIL_INDEX 1u

).";

	u32 bindSlot = 0;
	for (bindSlot; bindSlot < nofEdgeBuffers; ++bindSlot)
	{
		str << "layout(std430, binding = " << bindSlot << ") restrict readonly  buffer _edges" << bindSlot << " { uint edgeIndices" << bindSlot << "[]; };\n";
	}

	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _edgesInfos { vec4 edges[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _edgeRanges { uint edgeRanges[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _nofPotSilBuffers { uint nofPotSilJobs; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _indirectData { uint nofIndicesToDraw; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _computedMults { uint mults[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _clr { uint clr; };\n";

	str << R".(
layout (local_size_x = WG_SIZE) in;

uniform vec4 lightPosition;

).";
	if (useExtendedVersion)
	{
		str << R".(
uvec3 getEdgeRange(uint index)
{
	const uint pos = index * EDGERANGE_INFO_SIZE;
	return uvec3(edgeRanges[pos + 0], edgeRanges[pos + 1], edgeRanges[pos + 2]);
}
).";
	}
	else
	{
		str << R".(
uvec3 getEdgeRange(uint index)
{
	const uint pos = index * EDGERANGE_INFO_SIZE;
	return uvec3(edgeRanges[pos + 0], edgeRanges[pos + 1], 0);
}
).";
	}

	str << R".(
uint encodeEdgeMultiplicityToId(uint edgeID, int multiplicity)
{
	return uint(edgeID << MULTIPLICITY_BITS) | (multiplicity & MULTIPLICITY_MASK);
}
).";

	if (useExtendedVersion)
	{
		str << R".(
uint getEdgeId(uint pos, uint buf)
{	
).";
		for (u32 i = 0; i < nofEdgeBuffers; ++i)
		{
			str << "	if(buf== " << i << ") {return edgeIndices" << i << "[pos];}\n";
		}
		str << "	return 0;\n}\n";
	}
	else
	{
		str << R".(
uint getEdgeId(uint pos, uint unused)
{
	return edgeIndices0[pos];
}
).";
	}

	str << R".(

void pushEdge(uint storeIndex, uint encodedEdgeId)
{
	mults[storeIndex] = encodedEdgeId;
}

shared uint warpCounters[32]; //max 32 warps per WG on nV - 1024 threads
shared uint edgeStartLocal[WG_SIZE];
shared float tempData[WG_SIZE * 4 + 4]; //pre multiplicitu 2

uint decodeWorkSize(uint val)
{
	return val & 0x7FFFFFFF;
}

uint decodeIsSil(uint val)
{
	return uint((val & 0x80000000) !=0);
}

uint getPotEdgeStorePos(uint dataPerSide, uint warpId)
{
	if(gl_SubGroupInvocationARB == 0)
	{
		warpCounters[warpId] = 0;
	}
	
	const uint localOffset = atomicAdd(warpCounters[warpId], dataPerSide);
	
	uint globalOffset = 0;
	if(gl_SubGroupInvocationARB == 0)
	{
		globalOffset = atomicAdd(nofIndicesToDraw, warpCounters[warpId]);
	}
	
	globalOffset = readFirstInvocationARB(globalOffset);
	
	return globalOffset + localOffset;
}

uint getSilEdgeStorePos(uint localId, uint jobSize)
{
	if(localId == 0)
	{
		warpCounters[0] = atomicAdd(nofIndicesToDraw, jobSize);
	}

	barrier();

	return localId + warpCounters[0];
}

void load32B(uint localId, out vec4 d0, out vec4 d1)
{
	const uint thread_block_start = localId & 0xFFFFFFFE;
	const uint thread_block_offset = localId & 0x00000001;
	vec4 act_data[2];
	act_data[0] = edges[edgeStartLocal[thread_block_start + 0] + thread_block_offset];
	act_data[1] = edges[edgeStartLocal[thread_block_start + 1] + thread_block_offset];

	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].x;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].x;
	d0.x = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.x = tempData[thread_block_offset * WG_SIZE + localId + 1];
	
	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].y;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].y;
	d0.y = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.y = tempData[thread_block_offset * WG_SIZE + localId + 1];
	
	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].z;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].z;
	d0.z = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.z = tempData[thread_block_offset * WG_SIZE + localId + 1];
	
	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].w;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].w;
	d0.w = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.w = tempData[thread_block_offset * WG_SIZE + localId + 1];
}

void load64B(uint localId, out vec4 d0, out vec4 d1, out vec4 d2, out vec4 d3)
{
	const uint thread_block_start = localId & 0xFFFFFFFC;
	const uint thread_block_offset = localId & 0x00000003;
	vec4 act_data[4];
	act_data[0] = edges[edgeStartLocal[thread_block_start + 0] + thread_block_offset];
	act_data[1] = edges[edgeStartLocal[thread_block_start + 1] + thread_block_offset];
	act_data[2] = edges[edgeStartLocal[thread_block_start + 2] + thread_block_offset];
	act_data[3] = edges[edgeStartLocal[thread_block_start + 3] + thread_block_offset];

	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].x;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].x;
	tempData[localId + 2 * WG_SIZE + 2] = act_data[2].x;
	tempData[localId + 3 * WG_SIZE + 3] = act_data[3].x;
	d0.x = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.x = tempData[thread_block_offset * WG_SIZE + localId + 1];
	d2.x = tempData[thread_block_offset * WG_SIZE + localId + 2];
	d3.x = tempData[thread_block_offset * WG_SIZE + localId + 3];
	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].y;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].y;
	tempData[localId + 2 * WG_SIZE + 2] = act_data[2].y;
	tempData[localId + 3 * WG_SIZE + 3] = act_data[3].y;
	d0.y = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.y = tempData[thread_block_offset * WG_SIZE + localId + 1];
	d2.y = tempData[thread_block_offset * WG_SIZE + localId + 2];
	d3.y = tempData[thread_block_offset * WG_SIZE + localId + 3];
	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].z;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].z;
	tempData[localId + 2 * WG_SIZE + 2] = act_data[2].z;
	tempData[localId + 3 * WG_SIZE + 3] = act_data[3].z;
	d0.z = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.z = tempData[thread_block_offset * WG_SIZE + localId + 1];
	d2.z = tempData[thread_block_offset * WG_SIZE + localId + 2];
	d3.z = tempData[thread_block_offset * WG_SIZE + localId + 3];
	tempData[localId + 0 * WG_SIZE + 0] = act_data[0].w;
	tempData[localId + 1 * WG_SIZE + 1] = act_data[1].w;
	tempData[localId + 2 * WG_SIZE + 2] = act_data[2].w;
	tempData[localId + 3 * WG_SIZE + 3] = act_data[3].w;
	d0.w = tempData[thread_block_offset * WG_SIZE + localId + 0];
	d1.w = tempData[thread_block_offset * WG_SIZE + localId + 1];
	d2.w = tempData[thread_block_offset * WG_SIZE + localId + 2];
	d3.w = tempData[thread_block_offset * WG_SIZE + localId + 3];
}

void main()
{
	const uint wgId = gl_WorkGroupID.x;
	const uint localId = gl_LocalInvocationID.x;
	const uint warpId = localId / gl_SubGroupSizeARB;
	
	if(wgId >= nofPotSilJobs)
	{
		return;
	}
	
	if(gl_GlobalInvocationID.x == 0)
	{
		clr = 0;
	}

	const uvec3 job = getEdgeRange(wgId);
	const uint jobSize = decodeWorkSize(job.y);
	const uint jobStart = job.x;
	const uint isSil = decodeIsSil(job.y);
	const bool isActive = localId < jobSize;
	const uint actLocalId = (isActive || bool(isSil)) ? localId : (jobSize - 1); 

	if((localId >= ((jobSize + 3) & 0xFFFFFFFC)) || (bool(isSil) && !isActive))
	{
		return;
	}
	
	if(isSil==SIL_INDEX)
	{
		const uint edgeId = getEdgeId(jobStart + localId, job.z);
		uint storeIndex = getSilEdgeStorePos(localId, jobSize);
		
		pushEdge(storeIndex, edgeId);
	}
	else
	{
		const uint edgeId = getEdgeId(jobStart + actLocalId, job.z);
		const uint edgeStart = edgeId * NOF_VEC4_EDGE;
		
		vec4 data[MAX_MULTIPLICITY+2];
		vec4 v[2];
		
		edgeStartLocal[localId] = edgeStart;

		for(uint i = 0; i<BIG_POT_ITERS; ++i)
		{
			load64B(localId, data[4*i + 0], data[4*i + 1], data[4*i + 2], data[4*i + 3]);
		}

		for(uint i = 0; i<SMALL_POT_ITERS; ++i)
		{
			load32B(localId, data[4*BIG_POT_ITERS + 2*i + 0], data[4*BIG_POT_ITERS + 2*i + 1]);
		}

		int multiplicity = 0;
		
		for(uint ov = 0; ov < MAX_MULTIPLICITY; ++ov)
		{
			multiplicity += int(sign(dot(data[2+ov],lightPosition)));
		}
		
		const uint storeIndex = getPotEdgeStorePos(uint(multiplicity!=0)*NOF_DATA_PER_SIDE, warpId);
		
		if(multiplicity!=0)
		{
			pushEdge(storeIndex, encodeEdgeMultiplicityToId(edgeId, multiplicity));
		}
	}
}
).";

	return str.str();
}

std::string genSilExtrusionVs()
{
	return R".(
#version 450 core

in uint edgeId;

out flat uint oe;

void main()
{
	oe = edgeId;
}
).";
}

std::string genSilExtrusionGs(SidesGenShaderParams const& params)
{
	std::stringstream str;

	str << R".(
#version 450 core
).";
	str << "#define MULTIPLICITY_BITS " << params.nofBitsMultiplicity << "\n";
	str << "#define NOF_VEC4_EDGE " << params.edgeSizeNofVec4 << "u\n";

	str << R".(
#define MULTIPLICITY_MASK ((uint(1) << MULTIPLICITY_BITS) - 1)

layout(points)in;
).";

	str << "layout(triangle_strip,max_vertices= " << 4*params.maxMultiplicity <<")out;\n";

	str << R".(
layout(binding=0) restrict readonly buffer EdgeBuffer{vec4 edgeBuffer[];};

uniform mat4 mvp;
uniform vec4 lightPosition;

in flat uint oe[];

int decodeEdgeMultiplicityFromId(uint edgeWithEncodedMultiplicity)
{
	int retval = int(edgeWithEncodedMultiplicity & MULTIPLICITY_MASK);
	//sign extension
	int m = 1 << (MULTIPLICITY_BITS - 1);
	return (retval ^ m) - m;
}

uint decodeEdgeFromEncoded(uint edgeWithEncodedMultiplicity)
{
	return edgeWithEncodedMultiplicity >> MULTIPLICITY_BITS;
}

void swap(inout vec4 v1, inout vec4 v2)
{
	vec4 v3 = v1;
	v1 = v2;
	v2 = v3;
}

void main()
{
	const uint res = oe[0];
	const uint edgeId = decodeEdgeFromEncoded(res);
	const int multiplicity = decodeEdgeMultiplicityFromId(res);
	const uint absMultiplicity = uint(abs(multiplicity));
	vec4 P[4];

	P[0] = edgeBuffer[edgeId*NOF_VEC4_EDGE + 0];
	P[1] = edgeBuffer[edgeId*NOF_VEC4_EDGE + 1];
	P[2] = vec4(P[0].xyz*lightPosition.w-lightPosition.xyz,0);
	P[3] = vec4(P[1].xyz*lightPosition.w-lightPosition.xyz,0);
	P[0].w = P[1].w = 1.f;

	if(multiplicity > 0)
	{
		swap(P[0], P[1]);
		swap(P[2], P[3]);
	}

	gl_Position = mvp * P[0];EmitVertex();
	gl_Position = mvp * P[1];EmitVertex();
	gl_Position = mvp * P[2];EmitVertex();
	gl_Position = mvp * P[3];EmitVertex();
	EndPrimitive();
).";
	
	if(params.maxMultiplicity==2)
	{
		str << R".(
	if(absMultiplicity > 1)
	{
		gl_Position = mvp * P[0];EmitVertex();
		gl_Position = mvp * P[1];EmitVertex();
		gl_Position = mvp * P[2];EmitVertex();
		gl_Position = mvp * P[3];EmitVertex();
		EndPrimitive();
	}).";

	}
	else
	{
		str << R".(
	for(uint i = 1; i<absMultiplicity; ++i)
	{
		gl_Position = mvp * P[0];EmitVertex();
		gl_Position = mvp * P[1];EmitVertex();
		gl_Position = mvp * P[2];EmitVertex();
		gl_Position = mvp * P[3];EmitVertex();
		EndPrimitive();
	}).";
	}
	//*/
	str << "\n}";

	return str.str();
}
