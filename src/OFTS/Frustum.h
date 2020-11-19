#pragma once

#include "Plane.h"
#include <glm/glm.hpp>

class Frustum
{
public:

	Frustum();
	Frustum(float angle, float aspectRatio, float nearClipDist, float farClipDist, glm::vec3 const& pos, glm::vec3 const& focusPoint, glm::vec3 const& up);
	virtual ~Frustum() {}

	bool isFrustumIntersecting(Frustum const& frustum) const;
	bool TestSphere(glm::vec4 const& sphere) const;
	bool isLineIntersecting(glm::vec3 const& p0, glm::vec3 const& p1) const;

private:

	enum PlanePos
	{
		TOP = 0, 
		BOTTOM = 1, 
		LEFT = 2, 
		RIGHT = 3, 
		NEARP = 4, 
		FARP = 5,
		NOF_PLANES = 6
	};

	enum IntersectResult
	{ 
		OUTSIDE, 
		INTERSECT, 
		INSIDE 
	};

	//Planes & points
	oftsMath::Plane planes[NOF_PLANES];
	glm::vec3 ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr;
};