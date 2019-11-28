#pragma once

#include <Defines.h>

class Octree;

class CpuPotEdgePropagator
{
public:
	void propagateEdgesToUpperLevels(Octree* octree, bool isCompressed);

private:
	
	void PropagateEdgesToUpperLevelsCpu(u32 startingLevel);
	void ProcessEdgesInLevel(u32 levelNum);
	BitSet8 GetEdgeSyblingMask(u32 startingNodeID, u32 edgeID, u8 mask) const;
	void RemoveEdgeFromSyblings(u32 startingID, u32 edge, u8 mask, BitSet8 const& selectedSyblings);
	void AssignEdgeToNodeParent(u32 node, u32 edge, u8 mask);
	void AssignEdgeToNode(u32 node, u32 edge, u8 mask);

	void SortNodesInLevel(s32 level);
private:

	Octree* octree;
	bool IsCompressed = true;
};
