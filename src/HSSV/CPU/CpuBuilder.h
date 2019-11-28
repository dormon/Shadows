#pragma once

#include <Plane.h>
#include <Defines.h>

#include <vector>
#include <stack>
#include <unordered_map>
#include <mutex>
#include <memory>

class Octree;
class Adjacency;

class CpuBuilder
{
public:
	void fillOctree(Octree* octree, Adjacency const* adjacency, u32 bitPerMultiplicity, bool IsCompressed);

private:

	struct ChildResults
	{
		BitSet8 potMask;
		std::unordered_map<s32, BitSet8> silMasks; //for each multiplicity
	};

	struct CurrentStatus
	{
		u32 currentNode;
		u8 currentLevel;
	};

	typedef std::stack<CurrentStatus> Stack;

	std::vector< std::vector<Plane>> createEdgePlanes(Adjacency const* adjacency);
	
	void processEdge(std::vector<Plane> const& edgePlanes, u32 edgeId, Adjacency const* ad);

	ChildResults testChildNodes(u32 firstChild, u32 edgeID, std::vector<Plane> const& edgePlanes, Adjacency const* adjacency);

	void processSilhouetteEdge(CurrentStatus const& status, u32 edgeId, std::unordered_map<s32, BitSet8> const& silMasks);
	void processPotentialEdge(CurrentStatus const& status, u32 edgeId, BitSet8 const& potMask, Stack& stack);

	void storeSilhouetteEdgesCompressed(std::unordered_map<s32, BitSet8> const& silMasks, u32 edgeId, u32 nodeId);
	void storeSilhouetteEdges(std::unordered_map<s32, BitSet8> const& silMasks, u32 edgeId, u32 nodeId);
	
	void storePotentialEdgesCompressed(BitSet8 const& potMask, u32 edgeId, u32 parentNode);
	void storePotentialEdges(BitSet8 const& potMask, u32 edgeId, u32 parentNode);

	void storePotentialEdge(u32 edgeId, u32 nodeId, u8 bitmask);
	void storeSilhouetteEdge(u32 edgeId, u32 nodeId, u8 bitmask);

	void pushPotNodesOnStack(BitSet8 const& mask, CurrentStatus const& status, Stack& stack);

	void propagatePotEdges();

	void createAllOctreeMasks();

	void removeEmptyMasks();

private:
	Octree* octree = nullptr;
	u32 NofBitsMultiplicity = 0;
	bool IsCompressed = true;

	std::unique_ptr<std::mutex[]> Mutexes;
};
