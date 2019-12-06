#pragma once

#include <ShadowMethod.h>

class OmnidirFrustumTracedShadows : public ShadowMethod
{
public:

    OmnidirFrustumTracedShadows(vars::Vars& vars);

    virtual ~OmnidirFrustumTracedShadows();

    virtual void create(glm::vec4 const& lightPosition,
        glm::mat4 const& viewMatrix,
        glm::mat4 const& projectionMatrix) override;

protected:
	bool IsConservativeRasterizationSupported() const;

	void createBuffers();
	void createTriangleBuffer();
	void createVao();
	void createFbo();
	void createShaders();
	void createShadowMaskVao();
	void createShadowMaskFbo();
	void updateConstants();

	void renderIzb();
	void createShadowMask();
	void preprocessFrusta();

	bool IsValid = false;
};