#pragma once

#include <ShadowMethod.h>

class ShadowMapping : public ShadowMethod
{
public:
	ShadowMapping(vars::Vars& vars);

	virtual ~ShadowMapping();

	virtual void create(
		glm::vec4 const&lightPosition,
		glm::mat4 const&viewMatrix,
		glm::mat4 const&projectionMatrix) override;

protected:
	void createShadowMap();
	void createFbo();
	void createVao();
	void createShadowMaskVao();
	void createShadowMaskFbo();
	void createPrograms();
	void createLightViewMatrix();
	void createLightProjMatrix();

	void renderShadowMap(glm::mat4 const& lightVP);
	void renderShadowMask(glm::mat4 const& lightVP);
	
	glm::vec2							_texelSize;

	glm::mat4							_lightViewMatrix;
	glm::mat4							_lightProjMatrix;
};
