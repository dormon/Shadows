#pragma once

#include <ShadowMethod.h>

// Dual-paraboloid shadow mapping
class DPSM : public ShadowMethod
{
public:
	DPSM(vars::Vars& vars);

	virtual ~DPSM();

	virtual void create(
		glm::vec4 const& lightPosition,
		glm::mat4 const& viewMatrix,
		glm::mat4 const& projectionMatrix) override;

protected:
	void createShadowMap();
	void createShadowMapFbo();
	void createShadowMapVao();
	void createShadowMaskVao();
	void createShadowMaskFbo();
	
	void createPrograms();
	void createDpsmFillProgram();
	void createDpsmShadowProgram();

	void createLightViewMatrix();

	void renderShadowMap();
	void renderShadowMask();

	glm::mat4							_lightViewMatrix;
};
