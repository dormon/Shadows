#pragma once

#include <ShadowMethod.h>

class FrustumTracedShadows : public ShadowMethod
{
public:
    FrustumTracedShadows(vars::Vars& vars);

    virtual ~FrustumTracedShadows();

	virtual void create( glm::vec4 const& lightPosition,
						 glm::mat4 const& viewMatrix,
						 glm::mat4 const& projectionMatrix) override;

protected:
    bool IsConservativeRasterizationSupported() const;

	void createBuffers();
	void createVao();
	void createFbo();
	void createShaders();
	void createShadowMaskVao();
	void createShadowMaskFbo();
	void createLightViewMatrix();
	void createLightProjMatrix();

	void renderIzb(glm::mat4 const& lightVP);
	void createShadowMask(glm::mat4 const& lightVP);

	bool _isValid = false;
	glm::mat4 _lightViewMatrix;
	glm::mat4 _lightProjMatrix;
};