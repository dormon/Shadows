#include "Frustum.h"

Frustum::Frustum(){}

Frustum::Frustum(float fovy, float aspectRatio, float nearClipDist, float farClipDist, glm::vec3 const& pos, glm::vec3 const& lookAt, glm::vec3 const& up)
{
	//Vyska a sirka orezovych rovin
	const float tang = tanf(fovy * 0.5f);
	const float nearHeight = nearClipDist * tang;
	const float nearWidth = nearHeight * aspectRatio;
	const float farHeight = farClipDist * tang;
	const float farWidth = farHeight * aspectRatio;

	//Z
	glm::vec3 Z = glm::normalize(lookAt - pos);

	// X = UP x Z
	glm::vec3 X = glm::normalize(glm::cross(Z, up));

	//Y = Z x X
	glm::vec3 Y = glm::normalize(glm::cross(X, Z));

	//Centers
	glm::vec3 nearClipCenter = pos + (Z * nearClipDist);
	glm::vec3 farClipCenter = pos + (Z * farClipDist);

	//Near plane points
	ntl = nearClipCenter + (Y * nearHeight) - (X * nearWidth);
	ntr = nearClipCenter + (Y * nearHeight) + (X * nearWidth);
	nbl = nearClipCenter - (Y * nearHeight) - (X * nearWidth);
	nbr = nearClipCenter - (Y * nearHeight) + (X * nearWidth);

	//Far plane points
	ftl = farClipCenter + (Y * farHeight) - (X * farWidth);
	ftr = farClipCenter + (Y * farHeight) + (X * farWidth);
	fbl = farClipCenter - (Y * farHeight) - (X * farWidth);
	fbr = farClipCenter - (Y * farHeight) + (X * farWidth);

	//Planes
	planes[TOP].SetFromPoints(ntl, ntr, ftl);
	planes[BOTTOM].SetFromPoints(nbr, nbl, fbr);
	planes[LEFT].SetFromPoints(nbl, ntl, fbl);
	planes[RIGHT].SetFromPoints(ntr, nbr, fbr);
	planes[NEARP].SetFromPoints(ntr, ntl, nbr);
	planes[FARP].SetFromPoints(ftl, ftr, fbl);
}

bool Frustum::isFrustumIntersecting(Frustum const& f) const
{
	bool intersectA = 
		(isLineIntersecting(f.nbr, f.fbr)) ||
		(isLineIntersecting(f.nbl, f.ntl)) ||
		(isLineIntersecting(f.fbl, f.nbl)) ||
		(isLineIntersecting(f.nbl, f.nbr)) ||
		(isLineIntersecting(f.fbr, f.fbl)) ||
		(isLineIntersecting(f.fbr, f.ftr)) ||
		(isLineIntersecting(f.nbr, f.ntr)) ||
		(isLineIntersecting(f.fbl, f.ftl)) ||
		(isLineIntersecting(f.ntl, f.ftl)) ||
		(isLineIntersecting(f.ftl, f.ftr)) ||
		(isLineIntersecting(f.ftr, f.ntr)) ||
		(isLineIntersecting(f.ntr, f.ntl));

	bool intersectB = (f.isLineIntersecting(fbl, nbl)) ||
		(f.isLineIntersecting(nbl, nbr)) ||
		(f.isLineIntersecting(nbr, fbr)) ||
		(f.isLineIntersecting(fbr, fbl)) ||
		(f.isLineIntersecting(fbr, ftl)) ||
		(f.isLineIntersecting(nbr, ntr)) ||
		(f.isLineIntersecting(fbl, ftl)) ||
		(f.isLineIntersecting(nbl, ntl)) ||
		(f.isLineIntersecting(ntl, ftl)) ||
		(f.isLineIntersecting(ftl, ntr)) ||
		(f.isLineIntersecting(ftr, ntr)) ||
		(f.isLineIntersecting(ntr, ntl));

	return intersectA || intersectB;
}

bool Frustum::TestSphere(glm::vec4 const& sphere) const
{
	const float radius = sphere.w;
	int result = INSIDE;

	for (int i = 0; i < 6; i++)
	{
		float distance = planes[i].DistancePoint(glm::vec3(sphere));
		if (distance > radius)
		{
			return OUTSIDE;
		}
		else if (abs(distance) < radius)
		{
			result = INTERSECT;
		}
	}

	return result;
}

bool Frustum::isLineIntersecting(glm::vec3 const& p0, glm::vec3 const& p1) const
{
	const float fFrustumEdgeFudge = 0.01f;
	unsigned char outCode[2] = { 0, 0 };

	for (int i = 0; i < 6; i++)
	{
		float aDist = planes[i].DistancePoint(p0);
		float bDist = planes[i].DistancePoint(p1);
		int aSide = (aDist >= 0) ? 1 : 0;
		int bSide = (bDist >= 0) ? 1 : 0;

		outCode[0] |= (aSide << i);  // outcode stores the plane sidedness for all 6 clip planes
		outCode[1] |= (bSide << i);  // if outCode[0]==0, vertex is inside frustum

		if (aSide & bSide)   // trivial reject test (both points outside of one plane)
		{
			return false;
		}

		if (aSide ^ bSide)      // complex intersection test
		{
			glm::vec3 rayDir = p1 - p0;

			float t = planes[i].IntersectRay(p0, rayDir);
			if ((t >= 0.f) && (t <= 1.f))
			{
				glm::vec3 hitPt = p0 + t * rayDir;
				glm::vec4 sphere = glm::vec4(hitPt, fFrustumEdgeFudge);

				if (TestSphere(sphere) != OUTSIDE)
				{
					return true;
				}
			}
		}
	}

	return (outCode[0] == 0) || (outCode[1] == 0);
}
