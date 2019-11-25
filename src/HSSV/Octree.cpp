#include <Octree.h>
#include <MathOps.h>

#include <stack>
#include <iostream>
#include <string>


Octree::Octree(u32 deepestLevel, const AABB& volume)
{
	DeepestLevel = deepestLevel;

	GenerateLevelSizes();

	Init(volume);
}

void Octree::GenerateLevelSizes()
{
	LevelSizesInclusiveSum.clear();

	u32 prefixSum = 0;

	for (u32 i = 0; i <= DeepestLevel; ++i)
	{
		const u32 levelSize = MathOps::ipow(OCTREE_NUM_CHILDREN, i);
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

Node* Octree::getNode(u32 nodeID)
{
	if (nodeID > getTotalNumNodes() || !Nodes[nodeID].volume.isValid())
	{
		return nullptr;
	}

	return &(Nodes[nodeID]);
}

const Node* Octree::getNode(u32 nodeID) const
{
	if (nodeID > getTotalNumNodes() || !Nodes[nodeID].volume.isValid())
	{
		return nullptr;
	}

	return &(Nodes[nodeID]);
}

AABB Octree::getNodeVolume(u32 nodeID) const
{
	assert(nodeExists(nodeID));
	
	const auto n = getNode(nodeID);

	if (n)
	{
		return n->volume;
	}

	return AABB();
}

s32 Octree::getNodeParent(u32 nodeID) const
{
	if (nodeID == 0)
	{
		return -1;
	}
	
	return int(floor((nodeID - 1.f) / OCTREE_NUM_CHILDREN));
}
/*
int Octree::getNodeRecursionLevel(u32 nodeID) const
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

int Octree::getNodeIdInLevel(u32 nodeID) const
{
	int level = getNodeRecursionLevel(nodeID);

	return level > 0 ? getNodeIdInLevel(nodeID, level) : -1;
}

int Octree::getNodeIdInLevel(u32 nodeID, u32 level) const
{
	return nodeID - getNumNodesInPreviousLevels(level);
}
*/
s32 Octree::GetCorrespondingChildIndexFromPoint(u32 nodeID, const glm::vec3& point) const
{
	const glm::vec3 centerPoint = getNodeVolume(nodeID).getCenter();

	s32 r = (point.x >= centerPoint.x) + 2 * (point.y >= centerPoint.y) + 4 * (point.z >= centerPoint.z);
	return r;
}

bool Octree::nodeExists(u32 nodeID) const
{
	return (nodeID < getTotalNumNodes()) && Nodes[nodeID].volume.isValid();
}

/*
bool Octree::childrenExist(u32 nodeID) const
{
	int startID = getChildrenStartingId(nodeID);

	if (startID < 0)
	{
		return false;
	}

	return nodeExists(startID);
}
*/
void Octree::splitNode(u32 nodeID)
{
	AABB nodeVolume = getNodeVolume(nodeID);

	const s32 startingIndex = getChildrenStartingId(nodeID);

	for (u32 i = 0; i < OCTREE_NUM_CHILDREN; ++i)
	{
		CreateChild(nodeVolume, startingIndex + i, i);
	}
}

s32 Octree::getChildrenStartingId(u32 nodeID) const
{
	return OCTREE_NUM_CHILDREN * nodeID + 1;
}

void Octree::CreateChild(const AABB& parentSpace, u32 newNodeId, u32 indexWithinParent)
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

s32 Octree::getNodeIndexWithinParent(u32 nodeID) const
{
	s32 const parent = getNodeParent(nodeID);

	if(parent<0)
	{
		return 0;
	}

	return getNodeIndexWithinParent(nodeID, parent);
}

s32 Octree::getNodeIndexWithinParent(u32 nodeID, u32 parent) const
{
	const int startID = getChildrenStartingId(parent);

	return nodeID - startID;
}

bool Octree::isPointInsideOctree(const glm::vec3& point) const
{
	return MathOps::testAabbPointIsInsideOrOn(Nodes[0].volume, point);
}

u32 Octree::getDeepestLevel() const
{
	return DeepestLevel;
}

u32 Octree::getTotalNumNodes() const
{
	return LevelSizesInclusiveSum[DeepestLevel];
}

s32 Octree::getLevelFirstNodeID(u32 level) const
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

s32 Octree::getNumNodesInLevel(u32 level) const
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

	return sizeof(u32) * numIndices;
}

void Octree::makeNodesFit()
{
	Nodes.shrink_to_fit();
}

void Octree::ExpandWholeOctree()
{
	std::stack<u32> nodeStack;
	std::stack<s32> levelStack;

	nodeStack.push(0);
	levelStack.push(0);

	s32 const deepestLevel = getDeepestLevel();

	while (!nodeStack.empty())
	{
		const u32 node = nodeStack.top();
		const s32 currentLevel = levelStack.top();
		
		nodeStack.pop();
		levelStack.pop();

		splitNode(node);

		if (currentLevel < (deepestLevel - 1))
		{
			const auto startingChild = getChildrenStartingId(node);
			for (s32 i = 0; i < OCTREE_NUM_CHILDREN; ++i)
			{
				nodeStack.push(startingChild + i);
				levelStack.push(currentLevel + 1);
			}
		}
	}

	makeNodesFit();
}

u32 Octree::getLevelSize(u32 level) const
{
	if (level > getDeepestLevel())
	{
		return 0;
	}

	return MathOps::ipow(OCTREE_NUM_CHILDREN, level);
}

u32 Octree::getNumNodesInPreviousLevels(s32 level) const
{
	const int l = level - 1;

	assert(l < s32(LevelSizesInclusiveSum.size()));

	if (l < 0 || l>s32(DeepestLevel))
	{
		return 0;
	}

	return LevelSizesInclusiveSum[l];
}


void Octree::printNodePathToRoot(s32 nodeId) const
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