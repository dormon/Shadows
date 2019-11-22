#include <Octree.h>
#include <MathOps.h>

#include <stack>
#include <iostream>
#include <string>


Octree::Octree(uint32_t deepestLevel, const AABB& volume)
{
	DeepestLevel = deepestLevel;

	GenerateLevelSizes();

	Init(volume);
}

void Octree::GenerateLevelSizes()
{
	LevelSizesInclusiveSum.clear();

	uint32_t prefixSum = 0;

	for (uint32_t i = 0; i <= DeepestLevel; ++i)
	{
		const uint32_t levelSize = MathOps::ipow(OCTREE_NUM_CHILDREN, i);
		LevelSizesInclusiveSum.push_back(levelSize + prefixSum);
		prefixSum += levelSize;
	}
}

void Octree::Init(const AABB& volume)
{
	Node n;
	n.volume = volume;

	Nodes.resize(getTotalNumNodes());
	Nodes[0] = n;

	ExpandWholeOctree();
}

Node* Octree::getNode(uint32_t nodeID)
{
	if (nodeID > getTotalNumNodes() || !Nodes[nodeID].volume.isValid())
	{
		return nullptr;
	}

	return &(Nodes[nodeID]);
}

const Node* Octree::getNode(uint32_t nodeID) const
{
	if (nodeID > getTotalNumNodes() || !Nodes[nodeID].volume.isValid())
	{
		return nullptr;
	}

	return &(Nodes[nodeID]);
}

AABB Octree::getNodeVolume(uint32_t nodeID) const
{
	assert(nodeExists(nodeID));
	
	const auto n = getNode(nodeID);

	if (n)
	{
		return n->volume;
	}

	return AABB();
}

int Octree::getNodeParent(uint32_t nodeID) const
{
	if (nodeID == 0)
	{
		return -1;
	}
	
	return int(floor((nodeID - 1.f) / OCTREE_NUM_CHILDREN));
}
/*
int Octree::getNodeRecursionLevel(uint32_t nodeID) const
{
	int level = 0;
	for(auto size : _levelSizesInclusiveSum)
	{
		if (nodeID < size)
			return level;

		++level;
	}

	return -1;
}

int Octree::getNodeIdInLevel(uint32_t nodeID) const
{
	int level = getNodeRecursionLevel(nodeID);

	return level > 0 ? getNodeIdInLevel(nodeID, level) : -1;
}

int Octree::getNodeIdInLevel(uint32_t nodeID, uint32_t level) const
{
	return nodeID - getNumNodesInPreviousLevels(level);
}
*/
int Octree::GetCorrespondingChildIndexFromPoint(uint32_t nodeID, const glm::vec3& point) const
{
	const glm::vec3 centerPoint = getNodeVolume(nodeID).getCenter();

	int r = (point.x >= centerPoint.x) + 2 * (point.y >= centerPoint.y) + 4 * (point.z >= centerPoint.z);
	return r;
}
/*
bool Octree::nodeExists(uint32_t nodeID) const
{
	return (nodeID < getTotalNumNodes()) && Nodes[nodeID].volume.isValid();
}

bool Octree::childrenExist(uint32_t nodeID) const
{
	int startID = getChildrenStartingId(nodeID);

	if (startID < 0)
	{
		return false;
	}

	return nodeExists(startID);
}
*/
void Octree::splitNode(uint32_t nodeID)
{
	AABB nodeVolume = getNodeVolume(nodeID);

	const int startingIndex = getChildrenStartingId(nodeID);

	for (uint32_t i = 0; i < OCTREE_NUM_CHILDREN; ++i)
	{
		CreateChild(nodeVolume, startingIndex + i, i);
	}
}

int Octree::getChildrenStartingId(uint32_t nodeID) const
{
	return OCTREE_NUM_CHILDREN * nodeID + 1;
}

void Octree::CreateChild(const AABB& parentSpace, uint32_t newNodeId, uint32_t indexWithinParent)
{
	assert(!nodeExists(newNodeId));

	Node n;
	glm::vec3 minPoint = parentSpace.getMin();

	glm::vec3 isInPlus;
	isInPlus.x = (indexWithinParent & 1u) != 0 ? 1.0f : 0.f;
	isInPlus.y = (indexWithinParent & 2u) != 0 ? 1.0f : 0.f;
	isInPlus.z = (indexWithinParent & 4u) != 0 ? 1.0f : 0.f;

	glm::vec3 parentHalfExtents = parentSpace.getDiagonal() / glm::vec3(2.0f);

	glm::vec3 minPointOffset = isInPlus * parentHalfExtents;

	minPoint += minPointOffset;
	glm::vec3 maxPoint = minPoint + parentHalfExtents;

	n.volume = AABB(minPoint, maxPoint);

	Nodes[newNodeId] = n;
}

int Octree::getNodeIndexWithinParent(uint32_t nodeID) const
{
	const int parent = getNodeParent(nodeID);

	assert(parent >= 0);

	return getNodeIndexWithinParent(nodeID, parent);
}

int Octree::getNodeIndexWithinParent(uint32_t nodeID, uint32_t parent) const
{
	const int startID = getChildrenStartingId(parent);

	return nodeID - startID;
}

bool Octree::IsPointInsideOctree(const glm::vec3& point) const
{
	return MathOps::testAabbPointIsInsideOrOn(Nodes[0].volume, point);
}

uint32_t Octree::getDeepestLevel() const
{
	return DeepestLevel;
}

uint32_t Octree::getTotalNumNodes() const
{
	return LevelSizesInclusiveSum[DeepestLevel];
}

int Octree::getLevelFirstNodeID(uint32_t level) const
{
	if (level > DeepestLevel)
	{
		return -1;
	}

	if (level == 0)
	{
		return 0;
	}

	return LevelSizesInclusiveSum[level - 1];
}

int Octree::getNumNodesInLevel(uint32_t level) const
{
	if (level > DeepestLevel)
	{
		return -1;
	}

	return MathOps::ipow(OCTREE_NUM_CHILDREN, level);
}

uint64_t Octree::getOctreeSizeBytes() const
{
	uint64_t numIndices = 0;

	for(const auto& node : Nodes)
	{
		for (const auto& item : node.edgesAlwaysCastMap)
		{
			numIndices += item.second.size();
		}

		for (const auto& item : node.edgesMayCastMap)
		{
			numIndices += item.second.size();
		}
	}

	return sizeof(uint32_t) * numIndices;
}

void Octree::makeNodesFit()
{
	Nodes.shrink_to_fit();
}

void Octree::ExpandWholeOctree()
{
	std::stack<uint32_t> nodeStack;
	std::stack<int32_t> levelStack;

	nodeStack.push(0);
	levelStack.push(0);

	int const deepestLevel = getDeepestLevel();

	while (!nodeStack.empty())
	{
		const uint32_t node = nodeStack.top();
		const int32_t currentLevel = levelStack.top();
		
		nodeStack.pop();
		levelStack.pop();

		splitNode(node);

		if (currentLevel < (deepestLevel - 1))
		{
			const auto startingChild = getChildrenStartingId(node);
			for (int i = 0; i < OCTREE_NUM_CHILDREN; ++i)
			{
				nodeStack.push(startingChild + i);
				levelStack.push(currentLevel + 1);
			}
		}
	}

	makeNodesFit();
}

uint32_t Octree::getLevelSize(uint32_t level) const
{
	if (level > getDeepestLevel())
	{
		return 0;
	}

	return MathOps::ipow(OCTREE_NUM_CHILDREN, level);
}

void Octree::printNodePathToRoot(int nodeId) const
{
	std::cout << "Node path: ";
	while (nodeId >= 0)
	{
		std::cout << std::to_string(nodeId);
		nodeId = getNodeParent(nodeId);
		if (nodeId >= 0)
		{
			std::cout << " -> ";
		}
	}

	std::cout << std::endl;
}