#pragma once

#include <AABB.h>
#include <Plane.h>

class Adjacency;

enum class EdgeSilhouetness : int
{
	EDGE_NOT_SILHOUETTE = 0,
	EDGE_POTENTIALLY_SILHOUETTE = 1,
	EDGE_IS_SILHOUETTE_PLUS = 2,
	EDGE_IS_SILHOUETTE_MINUS = 3
};

enum class TestResult : int
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

	int greaterVec(const glm::vec3& a, const glm::vec3& b);

	int computeMult(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, const glm::vec4& L);

	int currentMultiplicity(const glm::vec3& A, const glm::vec3& B, const glm::vec3& O, const glm::vec4& L);

	int calcEdgeMultiplicity(Adjacency* edges, size_t edgeIndex, const glm::vec3& lightPos);
	
	bool isEdgeSpaceAaabbIntersecting(const Plane& p1, const Plane& p2, const AABB& voxel);

	int ipow(int base, int exp);
}