#pragma once

#include <ShadowMethod.h>

//Frustum Traced Shadows
//https://research.nvidia.com/publication/frustum-traced-raster-shadows-revisiting-irregular-z-buffers

class FTS : public ShadowMethod
{
public:
	FTS(vars::Vars& vars);
	virtual ~FTS();

	virtual void create(glm::vec4 const& lightPosition,
		glm::mat4 const& viewMatrix,
		glm::mat4 const& projectionMatrix) override;

private:
	
	void CreateSampler();
	void CreateTextures();
	void CreateHeadTex();
	void CreateLinkedListTex();
	void ClearTextures();
	void CreateTexture2D(char const* name, uint32_t format, uint32_t resX, uint32_t resY);

	void CompileShaders();
	void CreateFillProgram();
	void CreateShadowMaskProgram();

	void CreateIzb(glm::mat4 const& vp, glm::mat4 const& lightVP);
	glm::mat4 CreateLightViewMatrix() const;
	glm::mat4 CreateLightProjMatrix() const;
	uint32_t GetNofWgsFill() const;

	void CreateShadowMask(glm::mat4 const& lightVP);
	void CreateShadowMaskVao();
	void CreateShadowMaskFbo();

	glm::uvec2 GetWindowSize() const;
	glm::uvec2 GetLightResolution() const;

private:

	bool IsValid = false;
};