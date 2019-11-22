#include <MathOps.h>
#include <FastAdjacency.h>
#include <AdjacencyWrapper.h>

#define EDGE_IS_SILHOUETTE(sihlouettness) (int(sihlouettness) >= 2)

bool MathOps::isInRange(float value, float min, float max)
{
	return (value >= min) && (value < max);
}

float MathOps::testPlanePoint(const Plane& plane, const glm::vec3& point)
{
	glm::vec4 const eq = plane.get();
	return eq.x * point.x + eq.y * point.y + eq.z * point.z + eq.w;
}

bool MathOps::testAabbPointIsInsideOrOn(const AABB& bbox, const glm::vec3& point)
{
	const auto minPoint = bbox.getMin();
	const auto maxPoint = bbox.getMax();

	return isInRange(point.x, minPoint.x, maxPoint.x) && isInRange(point.y, minPoint.y, maxPoint.y) && isInRange(point.z, minPoint.z, maxPoint.z);
}

TestResult MathOps::interpretResult(float result)
{
	TestResult r = TestResult::INTERSECTS_ON;

	if (result > 0)
		r = TestResult::ABOVE_OUTSIDE;
	else if (result < 0)
		r = TestResult::BELOW_INSIDE;

	return r;
}

TestResult MathOps::testAabbPlane(const AABB& bbox, const Plane& plane)
{
	std::vector<glm::vec3> points = bbox.getVertices();

	const TestResult result = interpretResult(testPlanePoint(plane, points[0]));

	for (uint32_t i = 1; i < 8; ++i)
	{
		TestResult r = interpretResult(testPlanePoint(plane, points[i]));

		if (r != result)
			return TestResult::INTERSECTS_ON;
	}

	return result;
}

int MathOps::greaterVec(const glm::vec3& a, const glm::vec3& b)
{
	return int(glm::dot(glm::vec3(glm::sign(a - b)), glm::vec3(4, 2, 1)));
}

int MathOps::computeMult(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, const glm::vec4& L)
{
	glm::vec3 n = glm::cross(C - A, glm::vec3(L) - A * L.w);
	return int(glm::sign(dot(n, B - A)));
}

int MathOps::currentMultiplicity(const glm::vec3& A, const glm::vec3& B, const glm::vec3& O, const glm::vec4& L)
{
	if (greaterVec(A, O) > 0)
		return computeMult(O, A, B, L);
	else if (greaterVec(B, O) > 0)
		return -computeMult(A, O, B, L);
	else
		return computeMult(A, B, O, L);
}

int MathOps::calcEdgeMultiplicity(Adjacency const* edges, size_t edgeIndex, const glm::vec3& lightPos)
{
	glm::vec3 const& lowerPoint = getEdgeVertexLow(edges, edgeIndex);
	glm::vec3 const& higherPoint = getEdgeVertexHigh(edges, edgeIndex);
	
	const size_t nofOpposites = edges->getNofOpposite(edgeIndex);
	int multiplicity = 0;
	const glm::vec4 L = glm::vec4(lightPos, 1);

	for (size_t i = 0; i < nofOpposites; ++i)
	{
		multiplicity += currentMultiplicity(lowerPoint, higherPoint, getOppositeVertex(edges, edgeIndex, i), L);
	}

	return multiplicity;
}

bool MathOps::isEdgeSpaceAaabbIntersecting(std::vector<Plane> const& planes, const AABB& voxel)
{
	for(auto const& plane : planes)
	{
		if(testAabbPlane(voxel, plane) == TestResult::INTERSECTS_ON)
		{
			return true;
		}
	}
	
	return false;
}

int MathOps::ipow(int base, int exp)
{
	int result = 1;
	while (exp)
	{
		if (exp & 1)
		{
			result *= base;
		}

		exp >>= 1;
		base *= base;
	}

	return result;
}