#pragma once

#include <Node.h>

#include <vector>
#include <map>

//#define OCTREE_MAX_MULTIPLICITY 2

class Octree
{
public:

	Octree(u32 maxRecursionDepth, const AABB& volume);

	AABB getNodeVolume(u32 index) const;

	s32 getNodeParent(u32 nodeID) const;
	s32 getChildrenStartingId(u32 nodeID) const;
	s32 getNodeIndexWithinParent(u32 nodeID) const;
	s32 getNodeIndexWithinParent(u32 nodeID, u32 parent) const;

	u32 getNumNodesInPreviousLevels(s32 level) const;
	/*
	s32 getNodeRecursionLevel(u32 nodeID) const;
	s32 getNodeIdInLevel(u32 nodeID) const;
	s32 getNodeIdInLevel(u32 nodeID, u32 level) const;
	std::vector<u32> getLevelSizeInclusiveSum() const { return _levelSizesInclusiveSum; }
	*/
	void splitNode(u32 nodeID);

	Node* getNode(u32 nodeID);
	const Node* getNode(u32 nodeID) const;

	bool nodeExists(u32 nodeID) const;

	u32 getDeepestLevel() const;
	u32 getTotalNumNodes() const;
	s32 getLevelFirstNodeID(u32 level) const;
	s32 getNumNodesInLevel(u32 level) const;

	uint64_t getOctreeSizeBytes() const;

	void makeNodesFit();

	u32 getLevelSize(u32 level) const;

	void printNodePathToRoot(s32 nodeId) const;

	bool isPointInsideOctree(const glm::vec3& point) const;

private:

	void GenerateLevelSizes();
	void Init(const AABB& volume);
	void ExpandWholeOctree();

	void CreateChild(const AABB& parentSpace, u32 childID, u32 indexWithinParent);
	s32  GetCorrespondingChildIndexFromPoint(u32 nodeID, const glm::vec3& point) const;

	std::vector<Node> Nodes;

	std::vector<u32> LevelSizesInclusiveSum;

	u32 DeepestLevel;
};