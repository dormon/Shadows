#pragma once

#include <Defines.h>

#include <glm/glm.hpp>

class Octree;

class SidesDrawerBase
{
public:

	SidesDrawerBase(Octree* octree);
	virtual ~SidesDrawerBase() {}

	virtual void drawSides(const glm::mat4& mvp, const glm::vec4& light) = 0;

protected:
	s32 GetLowestLevelCellPoint(glm::vec3 const& pos);

protected:
	Octree* octree;
};