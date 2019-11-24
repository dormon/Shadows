#include <CPU/CpuPotEdgePropagator.h>
#include <Octree.h>
#include <MathOps.h>

//#include "HighResolutionTimer.hpp"
//#include <iostream>
#include <omp.h>
#include <algorithm>

void CpuPotEdgePropagator::propagateEdgesToUpperLevels(Octree* octree, bool IsCompressed)
{
	this->octree = octree;
	IsCompressed = IsCompressed;

	//HighResolutionTimer t;
	//t.reset();
	s32 const startingLevel = octree->getDeepestLevel() - 1;
	PropagateEdgesToUpperLevelsCpu(startingLevel);
	//auto dt = t.getElapsedTimeFromLastQueryMilliseconds();

	//std::cout << "Propagate Potential edges took " << dt / 1000.0f << " sec\n";
	//t.reset();
}

void CpuPotEdgePropagator::PropagateEdgesToUpperLevelsCpu(u32 startingLevel)
{
	for (s32 level = startingLevel; level > 0; --level)
	{
		ProcessEdgesInLevel(level);
		
		if(level>1)
		{
			SortNodesInLevel(level - 1);
		}
	}
}

void CpuPotEdgePropagator::ProcessEdgesInLevel(u32 level)
{
	assert(level > 0);
	s32 const startingID = octree->getLevelFirstNodeID(level);

	assert(startingID >= 0);

	s32 const stopId = MathOps::ipow(OCTREE_NUM_CHILDREN, level) + startingID;

	s32 currentID = startingID;

	#pragma omp parallel for 
	for (currentID = startingID; currentID<stopId; currentID += OCTREE_NUM_CHILDREN)
	{
		Node* firstNode = octree->getNode(u32(currentID));
		NodeMap const& edgesMap = firstNode->edgesMayCastMap;

		if (firstNode->edgesMayCastMap.find(BITMASK_ALL_SET) == edgesMap.end())
		{
			continue;
		}

		for (auto const edge : firstNode->edgesMayCastMap[BITMASK_ALL_SET])
		{
			BitSet8 const result = GetEdgeSyblingMask(currentID, edge, BITMASK_ALL_SET);

			if(IsCompressed)
			{
				if (result.count() > 1)
				{
					AssignEdgeToNodeParent(currentID, edge, u8(result.to_ulong()));
					RemoveEdgeFromSyblings(currentID, edge, BITMASK_ALL_SET, result);
				}
			}
			else
			{
				if(result.all())
				{
					AssignEdgeToNodeParent(currentID, edge, BITMASK_ALL_SET);
					RemoveEdgeFromSyblings(currentID, edge, BITMASK_ALL_SET, result);
				}
			}
		}
	}
}

BitSet8 CpuPotEdgePropagator::GetEdgeSyblingMask(u32 startingNodeID, u32 edgeID, u8 mask) const
{
	std::bitset<OCTREE_NUM_CHILDREN> retval;

	for (u32 i = 0; i < OCTREE_NUM_CHILDREN; ++i)
	{
		Node const* node = octree->getNode(startingNodeID + i);

		assert(node != nullptr);
		
		if (node->edgesMayCastMap.find(mask) != node->edgesMayCastMap.end() && std::binary_search(node->edgesMayCastMap.at(mask).begin(), node->edgesMayCastMap.at(mask).end(), edgeID))
		{
			retval.set(i);
		}
	}

	return retval;
}

void CpuPotEdgePropagator::AssignEdgeToNodeParent(u32 node, u32 edge, u8 mask)
{
	const int parent = octree->getNodeParent(node);

	assert(parent >= 0);

	AssignEdgeToNode(parent, edge, mask);
}

void CpuPotEdgePropagator::AssignEdgeToNode(u32 node, u32 edge, u8 mask)
{
	Node* n = octree->getNode(node);

	assert(n != nullptr);

	n->edgesMayCastMap[mask].push_back(edge);
}

void CpuPotEdgePropagator::SortNodesInLevel(s32 level)
{
	if(level<0)
	{
		return;
	}

	s32 const startingID = octree->getLevelFirstNodeID(level);
	s32 const endID = startingID + octree->getNumNodesInLevel(level);
	
	#pragma omp parallel for
	for(s32 nodeId = startingID; nodeId < endID; ++nodeId)
	{
		Node* n = octree->getNode(nodeId);

		assert(n != nullptr);

		if(n->edgesMayCastMap.find(BITMASK_ALL_SET) != n->edgesMayCastMap.end())
		{
			std::sort(n->edgesMayCastMap[BITMASK_ALL_SET].begin(), n->edgesMayCastMap[BITMASK_ALL_SET].end());
		}
	}
}

void CpuPotEdgePropagator::RemoveEdgeFromSyblings(u32 startingID, u32 edge, u8 mask, BitSet8 const& selectedSyblings)
{
	for (u32 i = 0; i<OCTREE_NUM_CHILDREN; ++i)
	{
		if(!selectedSyblings.test(i))
		{
			continue;
		}
		
		Node* node = octree->getNode(startingID + i);

		assert(node != nullptr);

		node->edgesMayCastMap[mask].erase(std::lower_bound(node->edgesMayCastMap[mask].begin(), node->edgesMayCastMap[mask].end(), edge));
	}
}