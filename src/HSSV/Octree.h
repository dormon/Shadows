#pragma once

#include <Node.h>

#include <vector>
#include <map>

#define OCTREE_NUM_CHILDREN 8
#define OCTREE_MAX_MULTIPLICITY 2


class Octree
{
public:

	Octree(uint32_t maxRecursionDepth, const AABB& volume);

	AABB getNodeVolume(uint32_t index) const;

	int getNodeParent(uint32_t nodeID) const;
	int getChildrenStartingId(uint32_t nodeID) const;
	int getNodeIndexWithinParent(uint32_t nodeID) const;
	int getNodeIndexWithinParent(uint32_t nodeID, uint32_t parent) const;
	/*
	int getNodeRecursionLevel(uint32_t nodeID) const;
	int getNodeIdInLevel(uint32_t nodeID) const;
	int getNodeIdInLevel(uint32_t nodeID, uint32_t level) const;
	std::vector<uint32_t> getLevelSizeInclusiveSum() const { return _levelSizesInclusiveSum; }
	*/
	void splitNode(uint32_t nodeID);

	Node* getNode(uint32_t nodeID);
	const Node* getNode(uint32_t nodeID) const;

	bool nodeExists(uint32_t nodeID) const;
	bool childrenExist(uint32_t nodeID) const;

	uint32_t getDeepestLevel() const;
	uint32_t getTotalNumNodes() const;
	int getLevelFirstNodeID(uint32_t level) const;
	int getNumNodesInLevel(uint32_t level) const;

	uint64_t getOctreeSizeBytes() const;

	void makeNodesFit();

	void setCompressionLevel(uint32_t ratio) { TreeCompressionLevel = ratio; }
	uint32_t getCompressionLevel() const { return TreeCompressionLevel; }
	uint32_t getLevelSize(uint32_t level) const;

	void printNodePathToRoot(int nodeId) const;

private:

	void GenerateLevelSizes();
	void Init(const AABB& volume);
	void ExpandWholeOctree();

	void CreateChild(const AABB& parentSpace, uint32_t childID, uint32_t indexWithinParent);
	int  GetCorrespondingChildIndexFromPoint(uint32_t nodeID, const glm::vec3& point) const;
	bool IsPointInsideOctree(const glm::vec3& point) const;

	std::vector<Node> Nodes;

	std::vector<uint32_t> LevelSizesInclusiveSum;

	uint32_t TreeCompressionLevel = 0;
	uint32_t DeepestLevel;
};