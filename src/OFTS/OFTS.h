#pragma once

#include <FTS/FTS.h>

#include "Frustum.h"

//Omnidirectional Frustum-Traced Shadows

class OFTS : public FTS
{
public:
	OFTS(vars::Vars& vars);

	virtual void create(glm::vec4 const& lightPosition,
		glm::mat4 const& viewMatrix,
		glm::mat4 const& projectionMatrix) override;

private:

	void ComputeLightFrusta();
	Frustum GetCameraFrustum(glm::mat4 const& viewMatrix) const;
	uint8_t GetUsedFrustumMasks(glm::mat4 const& viewMatrix) const;

	glm::mat4 GetLightViewMatrix(uint8_t index) const;
	glm::mat4 GetLightProjMatrix() const;

	void PrintStats(uint8_t mask) const;

	Frustum lightFrusta[6];
};