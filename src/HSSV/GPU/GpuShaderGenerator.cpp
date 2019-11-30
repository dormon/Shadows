#include <GPU/GpuShaderGenerator.h>

#include <sstream>

std::string getComputeEdgeRangesCsSource(std::vector<u32> const& lastNodePerBuffer, EdgeRangeShaderParams const& info)
{
	std::stringstream str;

	bool const useExtendedVersion = lastNodePerBuffer.size() > 1;
	u32 const numBuffers = u32(lastNodePerBuffer.size());

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
uniform uint nextKernelWgSize;

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
	uint nofWgs = 0;
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
			nofWgs = (nofEdges / nextKernelWgSize) + uint((nofEdges % nextKernelWgSize) != 0);
			storeIndex = atomicAdd(localAtomicCounter, nofWgs);
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
		for(uint i = 0; i< nofWgs; ++i)
		{
			const uint storeSize = (remainigEdges > nextKernelWgSize) ? nextKernelWgSize : remainigEdges;
).";
	if (useExtendedVersion)
	{
		str << "		storeSubbufferEdgeRange(edgesStart + i * nextKernelWgSize, encodeNofEdgesIsSil(storeSize, isSil), storeIndex +i, bufferNum);\n"; 
	}
	else
	{
		str << "		storeSubbufferEdgeRange(edgesStart + i * nextKernelWgSize, encodeNofEdgesIsSil(storeSize, isSil), storeIndex + i);\n";
	}

	str << R".(
			remainigEdges -= storeSize;
		}
	}
}
).";

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
	str << "#define MAX_OCTREE_LEVEL " << params.maxOctreeLevel << "u\n";
	str << "#define MULTIPLICITY_BITS " << params.nofBitsMultiplicity << "\n";
	str << "#define WG_SIZE " << params.wgSize << "u\n";

	str << R".(
#define NOF_VERTICES_PER_SIDE 6
#define MULTIPLICITY_MASK ((uint(1) << MULTIPLICITY_BITS) - 1)
#define POT_INDEX 0u
#define SIL_INDEX 1u

).";

	u32 bindSlot = 0;
	for(bindSlot; bindSlot < nofEdgeBuffers; ++bindSlot)
	{
		str << "layout(std430, binding = " << bindSlot << ") restrict readonly  buffer _edges" << bindSlot << " { uint edgeIndices" << bindSlot << "[]; };\n";
	}

	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _edgesInfos { float edges[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _opposingVerts { float oppositeVerts[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _edgeRanges { uint edgeRanges[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict readonly  buffer _nofPotSilBuffers { uint nofPotSilJobs; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _indirectData { uint nofIndicesToDraw; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _generatedSides { vec4 sides[]; };\n";
	str << "layout(std430, binding = " << bindSlot++ << ") restrict writeonly buffer _clr { uint clr; };\n";

	str << R".(
layout (local_size_x = WG_SIZE) in;

uniform vec4 lightPosition;
uniform uint nofEdges;

).";
	if(useExtendedVersion)
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

int decodeEdgeMultiplicityFromId(uint edgeWithEncodedMultiplicity)
{
	int retval = int(edgeWithEncodedMultiplicity & MULTIPLICITY_MASK);
	//sign extension
	int m = int(1u << (MULTIPLICITY_BITS - 1));
	return (retval ^ m) - m;
}

uint decodeEdgeFromEncoded(uint edgeWithEncodedMultiplicity)
{
	return edgeWithEncodedMultiplicity >> MULTIPLICITY_BITS;
}
).";

	if (useExtendedVersion)
	{
		str << R".(
uint getEdgeId(uint pos, uint buf)
{	
).";
		for(u32 i = 0; i< nofEdgeBuffers; ++i)
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
void getEdgePoints(in uint edgeId, inout vec3 lowerPoint, inout vec3 higherPoint)
{
	const uint lowOffset = 3*edgeId;
	lowerPoint = vec3(edges[lowOffset + 0], edges[lowOffset + 1], edges[lowOffset + 2]);
	
	const uint highOffset = 3*nofEdges + lowOffset;
	higherPoint = vec3(edges[highOffset + 0], edges[highOffset + 1], edges[highOffset + 2]);
}

vec3 getOppositeVertex(uint startingPos, uint oppositeVertexId)
{
	const uint offset = startingPos + 3*oppositeVertexId;
	return vec3(oppositeVerts[offset + 0], oppositeVerts[offset + 1], oppositeVerts[offset + 2]);
}

uint getOppositeVertexStartIndex(uint edgeId)
{
	return floatBitsToUint(edges[6*nofEdges + edgeId]);
}

uint getOppositeVertexCount(uint edgeId)
{
	return floatBitsToUint(edges[7*nofEdges + edgeId]);
}

void pushEdge(vec3 edgeVertices[2], uint startingIndex, int multiplicity)
{
	const uint swapVertices = uint(multiplicity>0);
	const uint absMultiplicity = abs(multiplicity);
	
	for(uint m = 0; m<absMultiplicity; ++m)
	{
		sides[startingIndex + m*NOF_VERTICES_PER_SIDE + 0] = vec4(edgeVertices[0^swapVertices], 1);
		sides[startingIndex + m*NOF_VERTICES_PER_SIDE + 1] = vec4(edgeVertices[1^swapVertices] - lightPosition.xyz, 0);
		sides[startingIndex + m*NOF_VERTICES_PER_SIDE + 2] = vec4(edgeVertices[0^swapVertices] - lightPosition.xyz, 0);
		sides[startingIndex + m*NOF_VERTICES_PER_SIDE + 3] = vec4(edgeVertices[1^swapVertices] - lightPosition.xyz, 0);
		sides[startingIndex + m*NOF_VERTICES_PER_SIDE + 4] = vec4(edgeVertices[0^swapVertices], 1);
		sides[startingIndex + m*NOF_VERTICES_PER_SIDE + 5] = vec4(edgeVertices[1^swapVertices], 1);
	}
}

//------------------MULTIPLICITY------------------
int greaterVec(vec3 a,vec3 b)
{
	return int(dot(ivec3(sign(a-b)),ivec3(4,2,1)));
}

int computeMult(vec3 A,vec3 B,vec3 C,vec3 L)
{
	vec3 n=cross(C-A,L-A);
	return int(sign(dot(n,B-A)));
}

int currentMultiplicity(vec3 A, vec3 B, vec3 O, vec3 L)
{
	if(greaterVec(A,O)>0)
		return computeMult(O,A,B,L);
	
	if(greaterVec(B,O)>0)
		return -computeMult(A,O,B,L);
	
	return computeMult(A,B,O,L);
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

uint getEdgeStorePos(uint absMultiplicity, uint warpId)
{
	if(gl_SubGroupInvocationARB == 0)
	{
		warpCounters[warpId] = 0;
	}
	
	const uint localOffset = atomicAdd(warpCounters[warpId], NOF_VERTICES_PER_SIDE * absMultiplicity);
	
	uint globalOffset = 0;
	if(gl_SubGroupInvocationARB == 0)
	{
		globalOffset = atomicAdd(nofIndicesToDraw, warpCounters[warpId]);
	}
	
	globalOffset = readFirstInvocationARB(globalOffset);
	
	return globalOffset + localOffset;
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
		const uint currentEdge = decodeEdgeFromEncoded(edgeId);
		const int multiplicity = decodeEdgeMultiplicityFromId(edgeId);
		
		vec3 points[2];
		getEdgePoints(currentEdge, points[0], points[1]);
		uint storeIndex = getEdgeStorePos(abs(multiplicity), warpId);
		
		pushEdge(points, storeIndex, multiplicity);
	}
	else
	{
		const uint edgeId = getEdgeId(jobStart + localId, job.z);
		
		vec3 points[2];
		getEdgePoints(edgeId, points[0], points[1]);
		
		int multiplicity = 0;
		const uint nofOpposite = getOppositeVertexCount(edgeId);
		const uint oppositeStartPos = getOppositeVertexStartIndex(edgeId);
		
		for(uint opposite = 0; opposite < nofOpposite; ++opposite)
		{
			vec3 oppositeVertex = getOppositeVertex(oppositeStartPos, opposite);
			multiplicity += currentMultiplicity(points[0], points[1], oppositeVertex, lightPosition.xyz);
		}
		
		const uint storeIndex = getEdgeStorePos(abs(multiplicity), warpId);
		
		if(multiplicity!=0)
		{
			pushEdge(points, storeIndex, multiplicity);
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

