#pragma once

#include <glm/glm.hpp>

namespace oftsMath
{
	class Plane
	{
	public:

		enum class IntersectResult
		{
			INTERSECT,
			NO_INTERSECT
		};

		Plane();
		Plane(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3);
		virtual ~Plane();

		void SetFromPoints(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3);
		void CreateFromPointNormalCCW(const glm::vec3& point, const glm::vec3& normal);

		float DistancePoint(const glm::vec3& point) const;

		IntersectResult TestLine(const glm::vec3& v1, const glm::vec3& v2) const;
		float IntersectRay(const glm::vec3& origin, const glm::vec3 dir) const;

		glm::vec4 GetPlane() const { return equation; }

	private:
		void toHessianNormalForm();

		glm::vec4 equation;
	};
};