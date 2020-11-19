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

protected:
	bool IsValid() const { return isValid; }

	void CreateTextures();
	void CreateHeatMap();
	void CreateHeadTex();
	void CreateLinkedListTex();
	void CreateMaxDepthTex();
	
	void CreateTexture2D(char const* name, uint32_t format, uint32_t resX, uint32_t resY);
	void CreateTexture2DArray(char const* name, uint32_t format, uint32_t resX, uint32_t resY, uint32_t depth);

	void ClearTextures();
	void ClearShadowMask();

	void CreateBuffers();
	void CreateMatrixBuffer();

	void CompileShaders(bool isOmnidirectional = false);
	void CreateHeatmapProgram(bool isOmnidirectional);
	void CreateMatrixProgram();
	void CreateIzbFillProgram();
	void CreateZbufferFillProgram();
	void CreateShadowMaskProgram();

	void ComputeHeatMap(glm::mat4 const& lightVP);

	void ComputeViewProjectionMatrix();
	glm::vec4 GetLightFrustumNearParams() const;

	void ComputeIzb(glm::mat4 const& vp, glm::mat4 const& lightV);

	uint32_t GetNofWgsFill() const;

	void InitShadowMaskZBuffer();
	void CreateDummyVao();

	void FillShadowMask(glm::mat4 const& lightV);
	void CreateShadowMaskVao();
	void CreateShadowMaskFbo();

	glm::uvec2 GetWindowSize() const;
	glm::uvec2 GetLightResolution() const;
	glm::uvec2 GetHeatmapResolution() const;

	glm::mat4 CreateLightViewMatrix() const;
	glm::mat4 CreateLightProjMatrix() const;

private:

	bool isValid = false;
};