#pragma once

#include <glm/glm.hpp>

class Plane
{
public:
	void createFromPointsCCW(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3);
	void createFromPointNormalCCW(const glm::vec3& point, const glm::vec3& normal);

	glm::vec4 get() const;

private:
	void toHessianNormalForm();

	glm::vec4 equation;
};