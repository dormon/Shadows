#pragma once

#include <ShadowMethod.h>
#include "Frustum.h"

//Omnidirectional Frustum-Traced Shadows
//TODO - one day, merge with FTS, most of the code is the same...
class OFTS: public ShadowMethod
{
public:
	OFTS(vars::Vars& vars);
	virtual ~OFTS();

	virtual void create(glm::vec4 const& lightPosition,
		glm::mat4 const& viewMatrix,
		glm::mat4 const& projectionMatrix) override;

private:

	void ComputeLightFrusta();
	Frustum GetCameraFrustum(glm::mat4 const& viewMatrix) const;
	uint8_t GetUsedFrustumMasks(glm::mat4 const& viewMatrix) const;

	glm::mat4 GetLightViewMatrix(uint8_t index, glm::vec3 const& lightPos) const;
	glm::mat4 GetLightProjMatrix() const;
	void CreateLightMatrices();

	void PrintStats(uint8_t mask) const;

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

	void CompileShaders();
	void CreateHeatmapProgram();
	void CreateMatrixProgram();
	void CreateIzbFillProgram();
	void CreateZbufferFillProgram();
	void CreateShadowMaskProgram();

	void ComputeHeatMap(uint8_t frustumMask);

	void ComputeViewProjectionMatrices(uint8_t frustumMask);
	glm::vec4 GetLightFrustumNearParams() const;

	void ComputeIzb();

	uint32_t GetNofWgsFill() const;

	void InitShadowMaskZBuffer();
	void CreateDummyVao();

	void FillShadowMask();
	void CreateShadowMaskVao();
	void CreateShadowMaskFbo();

	glm::uvec2 GetWindowSize() const;
	glm::uvec2 GetLightResolution() const;
	glm::uvec2 GetHeatmapResolution() const;
	glm::vec3  GetLightPosition() const;

private:

	bool isValid = false;
	Frustum lightFrusta[6];
	glm::mat4 lightV[6];
	glm::mat4 lightVP[6];
};