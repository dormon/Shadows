#pragma once

#include <ShadowMethod.h>

//Omnidirectional Frustum Traced Shadows
//https://research.nvidia.com/publication/frustum-traced-raster-shadows-revisiting-irregular-z-buffers

class OFTS : public ShadowMethod
{
public:
	OFTS(vars::Vars& vars);
	virtual ~OFTS();

	virtual void create(glm::vec4 const& lightPosition,
		glm::mat4 const& viewMatrix,
		glm::mat4 const& projectionMatrix) override;

private:
	
	void CreateHeadBuffer();
	void CreateLinkedListBuffer();
	void ClearBuffers();
	void CompileShaders();

	void CreateIzb();
	void CreateShadowMask();

private:
	bool IsValid = false;
};