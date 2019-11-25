#include <MathOps.h>
#include <FastAdjacency.h>
#include <AdjacencyWrapper.h>

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

	for (u32 i = 1; i < 8; ++i)
	{
		TestResult r = interpretResult(testPlanePoint(plane, points[i]));

		if (r != result)
			return TestResult::INTERSECTS_ON;
	}

	return result;
}

s32 MathOps::greaterVec(const glm::vec3& a, const glm::vec3& b)
{
	return int(glm::dot(glm::vec3(glm::sign(a - b)), glm::vec3(4, 2, 1)));
}

int MathOps::computeMult(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, const glm::vec4& L)
{
	glm::vec3 n = glm::cross(C - A, glm::vec3(L) - A * L.w);
	return int(glm::sign(dot(n, B - A)));
}

s32 MathOps::currentMultiplicity(const glm::vec3& A, const glm::vec3& B, const glm::vec3& O, const glm::vec4& L)
{
	if (greaterVec(A, O) > 0)
		return computeMult(O, A, B, L);
	else if (greaterVec(B, O) > 0)
		return -computeMult(A, O, B, L);
	else
		return computeMult(A, B, O, L);
}

s32 MathOps::calcEdgeMultiplicity(Adjacency const* edges, u32 edgeIndex, const glm::vec3& lightPos)
{
	glm::vec3 const& lowerPoint = getEdgeVertexLow(edges, edgeIndex);
	glm::vec3 const& higherPoint = getEdgeVertexHigh(edges, edgeIndex);
	
	size_t const nofOpposites = edges->getNofOpposite(edgeIndex);
	glm::vec4 const L = glm::vec4(lightPos, 1);
	s32 multiplicity = 0;

	for (size_t i = 0; i < nofOpposites; ++i)
	{
		multiplicity += currentMultiplicity(lowerPoint, higherPoint, getOppositeVertex(edges, edgeIndex, u32(i)), L);
		//multiplicity += computeMult(lowerPoint, higherPoint, getOppositeVertex(edges, edgeIndex, u32(i)), L);
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

s32 MathOps::ipow(s32 base, s32 exp)
{
	s32 result = 1;
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

s8 MathOps::findFirstSet(u8 num)
{
	if(num==0)
	{
		return -1;
	}

	u8 cnt = 0;
	while((num & 1)==0)
	{
		cnt++;
		num >>= 1;
	}

	return cnt;
}

u32 MathOps::getMaxNofSignedBits(u32 num)
{
	return u32(ceil(log2(num))) + 2;
}
