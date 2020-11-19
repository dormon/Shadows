#include "Plane.h"

using namespace oftsMath;

Plane::Plane(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
{
	SetFromPoints(v1, v2, v3);
}

Plane::Plane()
{
	equation = glm::vec4(0);
}

Plane::~Plane() {}

void Plane::SetFromPoints(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3)
{
	glm::vec3 normal = glm::normalize(glm::cross(glm::normalize(v3 - v2), glm::normalize(v1 - v2)));

	CreateFromPointNormalCCW(v2, normal);
}

void Plane::CreateFromPointNormalCCW(const glm::vec3& point, const glm::vec3& normal)
{
	equation = glm::vec4(normal, -point.x * normal.x - point.y * normal.y - point.z * normal.z);
	//toHessianNormalForm();
}

float Plane::DistancePoint(const glm::vec3& point) const
{
	return glm::dot(equation, glm::vec4(point, 1));
}

Plane::IntersectResult Plane::TestLine(const glm::vec3& v1, const glm::vec3& v2) const
{
	const float d0 = DistancePoint(v1);
	const float d1 = DistancePoint(v2);

	if ((d0 * d1) <= 0)
	{
		return IntersectResult::INTERSECT;
	}
	
	return IntersectResult::NO_INTERSECT;
}

float Plane::IntersectRay(const glm::vec3& origin, const glm::vec3 dir) const
{
	float distance = glm::dot(equation, glm::vec4(origin, 1));
	float len = glm::dot(glm::vec3(equation), dir);

	return -distance / len;
}

void Plane::toHessianNormalForm()
{
	equation = equation / (glm::sqrt(equation.x * equation.x + equation.y * equation.y + equation.z * equation.z));
}