#pragma once

#include <AABB.h>
#include <Plane.h>
#include <Defines.h>

class Adjacency;

enum class EdgeSilhouetness : s32
{
	EDGE_NOT_SILHOUETTE = 0,
	EDGE_POTENTIALLY_SILHOUETTE = 1,
	EDGE_IS_SILHOUETTE_PLUS = 2,
	EDGE_IS_SILHOUETTE_MINUS = 3
};

enum class TestResult : s32
{
	ABOVE_OUTSIDE = 1,
	BELOW_INSIDE = -1,
	INTERSECTS_ON = 0
};

namespace MathOps
{
	bool isInRange(float value, float min, float max);

	float testPlanePoint(const Plane& plane, const glm::vec3& point);

	bool testAabbPointIsInsideOrOn(const AABB& bbox, const glm::vec3& point);

	TestResult interpretResult(float result);

	TestResult testAabbPlane(const AABB& bbox, const Plane& plane);

	s32 greaterVec(const glm::vec3& a, const glm::vec3& b);

	s32 computeMult(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, const glm::vec4& L);

	s32 currentMultiplicity(const glm::vec3& A, const glm::vec3& B, const glm::vec3& O, const glm::vec4& L);

	s32 calcEdgeMultiplicity(Adjacency const* edges, u32 edgeIndex, const glm::vec3& lightPos);
	
	bool isEdgeSpaceAaabbIntersecting(std::vector<Plane> const& planes, const AABB& voxel);

	s32 ipow(s32 base, s32 exp);

	s8 findFirstSet(u8 num);

	u32 getMaxNofSignedBits(u32 num);
}