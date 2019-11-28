#include <SidesDrawerBase.h>
#include <Octree.h>
#include <MathOps.h>

SidesDrawerBase::SidesDrawerBase(Octree* o)
{
	octree = o;
}

s32 SidesDrawerBase::GetLowestLevelCellPoint(glm::vec3 const& point)
{
	if (!octree->isPointInsideOctree(point))
	{
		return -1;
	}

	AABB const& bb = octree->getNode(0)->volume;
	glm::vec3 const singleUnitSize = bb.getDiagonal() / glm::vec3(float(MathOps::ipow(2, octree->getDeepestLevel())));

	glm::uvec3 pos = glm::uvec3(glm::floor((point - bb.getMin()) / singleUnitSize));

	u32 const deepstLevel = octree->getDeepestLevel();
	s32 relPos = 0;

	for (uint32_t i = 0; i < deepstLevel; ++i)
	{
		relPos += ((((pos.x >> i) & 1) << 0) + (((pos.y >> i) & 1) << 1) + (((pos.z >> i) & 1) << 2))* MathOps::ipow(OCTREE_NUM_CHILDREN, i);
	}

	relPos += octree->getNumNodesInPreviousLevels(deepstLevel);

	return relPos;
}

