#include "OFTS.h"
#include "OFTS_shaderGen.h"

#include <geGL/Texture.h>
#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/Framebuffer.h>
#include <geGL/VertexArray.h>

#include <Model.h>
#include <Deferred.h>

#include <BasicCamera/FreelookCamera.h>
#include <ifExistStamp.h>
#include <FunctionPrologue.h>

#include <iostream>

using namespace ge;
using namespace gl;

constexpr const char* wgsizeParamName = "fts.args.wgSize";
constexpr const char* resolutionParamName = "fts.args.resolution";
constexpr const char* nearParamName = "fts.args.nearZ";
constexpr const char* farParamName = "fts.args.farZ";
constexpr const char* listTresholdParamName = "fts.args.longListTreshold";
constexpr const char* heatmapResParamName = "fts.args.heatmapRes";
constexpr const char* biasParamName = "fts.args.traversalBias";

constexpr const char* listTexName = "ofts.objects.listTex";
constexpr const char* headTexName = "ofts.objects.headTex";
constexpr const char* maxDepthTexName = "ofts.objects.maxDepthTex";
constexpr const char* heatmapTexName = "ofts.objects.heatmap";

constexpr const char* matrixBufferName = "ofts.objects.matrixBuffer";

constexpr const char* heatmapProgramName = "ofts.objects.heatProgram";
constexpr const char* matrixProgramName = "ofts.objects.matrixProgram";
constexpr const char* fillProgramName = "ofts.objects.fillProgram";
constexpr const char* zbufferProgramName = "ofts.objects.zbufferProgram";
constexpr const char* shadowMaskProgrammName = "ofts.objects.shadowMaskProgram";

constexpr const char* fboName = "ofts.objects.fboName";
constexpr const char* depthTexName = "ofts.objects.depthTex";
constexpr const char* shadowMaskVaoName = "ofts.objects.shadowMaskVao";
constexpr const char* dummyVaoName = "ofts.objects.dummyVao";

constexpr const GLuint heatmapTexFormat = GL_R32UI;
constexpr const GLuint listTexFormat = GL_R32I;
constexpr const GLuint headTexFormat = GL_R32I;
constexpr const GLuint maxDepthTexFormat = GL_R32F;

constexpr const float fovyAngleRad = 1.5708f; //90degs

glm::vec3 const g_Ups[] = {
		glm::vec3(0, 1, 0), //+x
		glm::vec3(0, 1, 0), //-x
		glm::vec3(0, 0, 1), //+y
		glm::vec3(0, 0,-1), //-y
		glm::vec3(0, 1, 0), //+z
		glm::vec3(0, 1, 0), //-z
};

glm::vec3 const g_Dirs[] = {
	glm::vec3(1, 0, 0), //+x
	glm::vec3(-1, 0, 0),//-x
	glm::vec3(0, 1, 0), //+y
	glm::vec3(0,-1, 0), //-y
	glm::vec3(0, 0, 1), //+z
	glm::vec3(0, 0,-1), //-z
};


OFTS::OFTS(vars::Vars& vars) : ShadowMethod(vars) 
{
	isValid = IsConservativeRasterizationSupported();
}

OFTS::~OFTS()
{
	vars.erase("ofts.objects");
}

void OFTS::ComputeLightFrusta()
{
	FUNCTION_PROLOGUE("fts.objects", "lightPosition", nearParamName, farParamName);

	const float nearZ = vars.getFloat("fts.args.nearZ");
	const float farZ = vars.getFloat("fts.args.farZ");
	const float fovy = fovyAngleRad;
	const float aspectRatio = 1.f;
	glm::vec3 const lightPos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));

	for(uint32_t i = 0; i<6; ++i)
	{
		lightFrusta[i] = Frustum(fovy, aspectRatio, nearZ, farZ, lightPos, lightPos + g_Dirs[i], g_Ups[i]);
	}
}

Frustum OFTS::GetCameraFrustum(glm::mat4 const& viewMatrix) const
{
	float const fovy = vars.getFloat("args.camera.fovy");
	float const near = vars.getFloat("args.camera.near");
	float far = vars.getFloat("args.camera.far");
	far = isinf(far) != 0 ? 50000.f : far;
	
	glm::vec2 const windowSize = glm::vec2(*vars.get<glm::uvec2>("windowSize"));
	const float aspectRatio = windowSize.x / windowSize.y;

	glm::vec3 const pos = vars.get<basicCamera::FreeLookCamera>("cameraTransform")->getPosition();
	glm::vec3 const up = glm::vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
	glm::vec3 const forward = -glm::vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);
	
	return Frustum(fovy, aspectRatio, near, far, pos, pos + forward, up);
}

uint8_t OFTS::GetUsedFrustumMasks(glm::mat4 const& viewMatrix) const
{
	uint8_t mask = 0;

	Frustum const camFrustum = GetCameraFrustum(viewMatrix);

	for(uint8_t i =0; i<6; ++i)
	{
		if(camFrustum.isFrustumIntersecting(lightFrusta[i]))
		{
			mask |= (uint8_t(1) << i);
		}
	}

	return mask;
}

glm::mat4 OFTS::GetLightViewMatrix(uint8_t index, glm::vec3 const& lightPos) const
{
	assert(index < 6);

	return glm::lookAt(lightPos, lightPos + g_Dirs[index], g_Ups[index]);
}

void OFTS::CreateLightMatrices()
{
	FUNCTION_PROLOGUE("ofts.objects", "lightPosition", nearParamName, farParamName);

	glm::vec3 const lightPos = GetLightPosition();
	glm::mat4 const lightP = GetLightProjMatrix();

	lightV[0] = GetLightViewMatrix(0, lightPos);
	lightV[1] = GetLightViewMatrix(1, lightPos);
	lightV[2] = GetLightViewMatrix(2, lightPos);
	lightV[3] = GetLightViewMatrix(3, lightPos);
	lightV[4] = GetLightViewMatrix(4, lightPos);
	lightV[5] = GetLightViewMatrix(5, lightPos);

	lightVP[0] = lightP * lightV[0];
	lightVP[1] = lightP * lightV[1];
	lightVP[2] = lightP * lightV[2];
	lightVP[3] = lightP * lightV[3];
	lightVP[4] = lightP * lightV[4];
	lightVP[5] = lightP * lightV[5];
}

glm::mat4 OFTS::GetLightProjMatrix() const
{
	const float nearZ = vars.getFloat("fts.args.nearZ");
	const float farZ = vars.getFloat("fts.args.farZ");
	const float fovy = fovyAngleRad;
	const float aspectRatio = 1.f;

	return glm::perspective(fovy, aspectRatio, nearZ, farZ);
}

void OFTS::PrintStats(uint8_t mask) const
{
	static uint8_t prevMask = 0;

	if (prevMask != mask)
	{
		prevMask = mask;
		int nofBits = 0;
		for (int b = 0; b < 6; ++b)
		{
			if ((mask >> b) & 1)
			{
				++nofBits;
			}
		}
		std::cout << uint32_t(mask) << " (" << nofBits << "): ";

		for (uint32_t i = 0; i < 6; ++i)
		{
			if ((mask >> i) & 1)
			{
				switch (i)
				{
				case 0: std::cout << "+x "; break;
				case 1: std::cout << "-x "; break;
				case 2: std::cout << "+y "; break;
				case 3: std::cout << "-y "; break;
				case 4: std::cout << "+z "; break;
				case 5: std::cout << "-z "; break;
				default:
					break;
				}
			}
		}

		std::cout << std::endl;
	}
}

void OFTS::CreateTextures()
{
	CreateHeatMap();
	CreateHeadTex();
	CreateLinkedListTex();
	CreateMaxDepthTex();
}

void OFTS::CreateHeatMap()
{
	FUNCTION_PROLOGUE("ofts.objects", heatmapResParamName);

	glm::uvec2 res = GetHeatmapResolution();
	CreateTexture2DArray(heatmapTexName, heatmapTexFormat, res.x, res.y, 6);
}

void OFTS::CreateHeadTex()
{
	FUNCTION_PROLOGUE("ofts.objects", resolutionParamName, "renderModel");

	glm::uvec2 const lightRes = GetLightResolution();
	CreateTexture2DArray(headTexName, headTexFormat, lightRes.x, lightRes.y, 12);
}

void OFTS::CreateLinkedListTex()
{
	FUNCTION_PROLOGUE("ofts.objects", "windowSize", "renderModel");

	glm::uvec2 const windowSize = GetWindowSize();
	CreateTexture2D(listTexName, listTexFormat, windowSize.x, windowSize.y);
}

void OFTS::CreateMaxDepthTex()
{
	FUNCTION_PROLOGUE("ofts.objects", resolutionParamName, "renderModel");

	glm::uvec2 const lightRes = GetLightResolution();
	CreateTexture2DArray(maxDepthTexName, maxDepthTexFormat, lightRes.x, lightRes.y, 12);
}

void OFTS::CreateTexture2D(char const* name, uint32_t format, uint32_t resX, uint32_t resY)
{
	vars.reCreate<Texture>(name, GL_TEXTURE_2D, format, 1, resX, resY);
}

void OFTS::CreateTexture2DArray(char const* name, uint32_t format, uint32_t resX, uint32_t resY, uint32_t depth)
{
	vars.reCreate<Texture>(name, GL_TEXTURE_2D_ARRAY, format, 1, resX, resY, depth);
}

void OFTS::ClearTextures()
{
	assert(glGetError() == GL_NO_ERROR);

	//-1
	int const clearVal = -1;
	vars.get<Texture>(listTexName)->clear(0, GL_RED_INTEGER, GL_INT, &clearVal);
	vars.get<Texture>(headTexName)->clear(0, GL_RED_INTEGER, GL_INT, &clearVal);

	//0
	vars.get<Texture>(maxDepthTexName)->clear(0, GL_RED, GL_FLOAT);
	vars.get<Texture>(heatmapTexName)->clear(0, GL_RED_INTEGER, GL_INT);

	//vars.get<Buffer>("xtmp")->clear(GL_R32F, GL_RED, GL_FLOAT);

	assert(glGetError() == GL_NO_ERROR);
}

void OFTS::ClearShadowMask()
{
	float const val = 1.f;
	vars.get<Texture>("shadowMask")->clear(0, GL_RED, GL_FLOAT, &val);
}

void OFTS::CreateBuffers()
{
	CreateMatrixBuffer();
	//vars.reCreate<Buffer>("xtmp", 20 * sizeof(float));
}

void OFTS::CreateMatrixBuffer()
{
	FUNCTION_PROLOGUE("ofts.objects", "renderModel");
	vars.reCreate<Buffer>(matrixBufferName, 12 * 16 * sizeof(float) + 6 * sizeof(uint32_t)); //12 matrices + 6 uints
}


void OFTS::CompileShaders()
{
	CreateHeatmapProgram();
	CreateMatrixProgram();
	CreateIzbFillProgram();
	CreateZbufferFillProgram();
	CreateShadowMaskProgram();
}

void OFTS::CreateHeatmapProgram()
{
	FUNCTION_PROLOGUE("ofts.objects", wgsizeParamName);
	OftsShaderGen gen;

	uint32_t const wgSize = vars.getUint32(wgsizeParamName);
	vars.reCreate<Program>(heatmapProgramName, gen.GetHeatmapCS(wgSize));
}

void OFTS::CreateMatrixProgram()
{
	FUNCTION_PROLOGUE("ofts.objects");

	OftsShaderGen gen;
	vars.reCreate<Program>(matrixProgramName, gen.GetMatrixCS());
}

void OFTS::CreateIzbFillProgram()
{
	FUNCTION_PROLOGUE("ofts.objects", wgsizeParamName);
	OftsShaderGen gen;

	uint32_t const wgSize = vars.getUint32(wgsizeParamName);
	vars.reCreate<Program>(fillProgramName, gen.GetIzbFillCS(wgSize));
}

void OFTS::CreateZbufferFillProgram()
{
	FUNCTION_PROLOGUE("ofts.objects");

	OftsShaderGen gen;
	vars.reCreate<Program>(zbufferProgramName, gen.GetZBufferFillVS(), gen.GetZBufferFillGS(), gen.GetZBufferFillFS());
}

void OFTS::CreateShadowMaskProgram()
{
	FUNCTION_PROLOGUE("ofts.objects");

	OftsShaderGen gen;
	vars.reCreate<Program>(shadowMaskProgrammName, gen.GetShadowMaskVS(), gen.GetShadowMaskGS(), gen.GetShadowMaskFS());
}

void OFTS::ComputeHeatMap(uint8_t frustumMask)
{
	assert(glGetError() == GL_NO_ERROR);

	Program* program = vars.get<Program>(heatmapProgramName);
	Texture* heatMap = vars.get<Texture>(heatmapTexName);
	Texture* shadowMask = vars.get<Texture>("shadowMask");

	glm::uvec2 const screenRes = GetWindowSize();
	glm::uvec2 const heatmapRes = GetHeatmapResolution();
	glm::vec3 const lightPos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));
	uint32_t const nofWgs = GetNofWgsFill();

	program->use();
	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP[0]), 6);
	program->set3fv("lightPos", glm::value_ptr(lightPos));
	program->set2uiv("screenResolution", glm::value_ptr(screenRes));
	program->set2uiv("heatmapResolution", glm::value_ptr(heatmapRes));
	program->set1ui("frustumMask", uint32_t(frustumMask));

	heatMap->bindImage(0);
	shadowMask->bindImage(1);
	vars.get<GBuffer>("gBuffer")->position->bind(0);
	vars.get<GBuffer>("gBuffer")->normal->bind(1);

	glDispatchCompute(nofWgs, 1, 1);

	heatMap->unbind(0);
	shadowMask->unbind(1);
	vars.get<GBuffer>("gBuffer")->position->unbind(0);
	vars.get<GBuffer>("gBuffer")->normal->unbind(1);

	assert(glGetError() == GL_NO_ERROR);
}

void OFTS::ComputeViewProjectionMatrices(uint8_t frustumMask)
{
	assert(glGetError() == GL_NO_ERROR);

	Buffer* matrixBuffer = vars.get<Buffer>(matrixBufferName);
	Texture* heatMap = vars.get<Texture>(heatmapTexName);
	Program* program = vars.get<Program>(matrixProgramName);

	glm::vec4 const frustumData = GetLightFrustumNearParams();
	glm::uvec2 const resolution = GetHeatmapResolution();
	uint32_t const treshold = vars.getUint32(listTresholdParamName);

	program->use();
	program->set4fv("frustumParams", glm::value_ptr(frustumData));
	program->set2uiv("heatmapResolution", glm::value_ptr(resolution));
	program->set1ui("treshold", treshold);
	program->set1ui("frustumMask", uint32_t(frustumMask));

	matrixBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, 0);

	heatMap->bindImage(0);

	glDispatchCompute(6, 1, 1);

	matrixBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	heatMap->unbind(0);

	assert(glGetError() == GL_NO_ERROR);
}

glm::vec4 OFTS::GetLightFrustumNearParams() const
{
	float const fovyHalf = 0.5f * fovyAngleRad;
	float const nearDist = vars.getFloat(nearParamName);
	float const farDist = vars.getFloat(farParamName);

	glm::vec2 res = glm::vec2(GetLightResolution());
	float const aspect = res.x / res.y;

	float w = 2.f * nearDist * glm::tan(fovyHalf);
	float h = w / aspect;

	glm::vec4 retVal;
	retVal.x = w;
	retVal.y = h;
	retVal.z = nearDist;
	retVal.w = farDist;

	return retVal;
}

void OFTS::ComputeIzb()
{
	assert(glGetError() == GL_NO_ERROR);

	Program* program = vars.get<Program>(fillProgramName);

	Texture* headTex = vars.get<Texture>(headTexName);
	Texture* listTex = vars.get<Texture>(listTexName);
	Texture* maxDepthTex = vars.get<Texture>(maxDepthTexName);

	Buffer* matrixBuffer = vars.get<Buffer>(matrixBufferName);

	program->use();

	glm::uvec2 const screenRes = GetWindowSize();
	glm::uvec2 const lightRes = GetLightResolution();
	glm::vec3 const lightPos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));

	program->set2uiv("screenResolution", glm::value_ptr(screenRes));
	program->set2uiv("lightResolution", glm::value_ptr(lightRes));
	program->set3fv("lightPos", glm::value_ptr(lightPos));
	program->setMatrix4fv("lightV", glm::value_ptr(lightV[0]), 6);

	headTex->bindImage(0, 0, headTexFormat, GL_READ_WRITE, GL_TRUE, 0);
	listTex->bindImage(1, 0, listTexFormat, GL_READ_WRITE);
	maxDepthTex->bindImage(2, 0, maxDepthTexFormat, GL_READ_WRITE, GL_TRUE, 0);

	vars.get<GBuffer>("gBuffer")->position->bind(0);
	vars.get<GBuffer>("gBuffer")->normal->bind(1);
	//vars.get<Buffer>("xtmp")->bindBase(GL_SHADER_STORAGE_BUFFER, 3);

	matrixBuffer->bindBase(GL_UNIFORM_BUFFER, 0);

	uint32_t const nofWgs = GetNofWgsFill();

	glDispatchCompute(nofWgs, 1, 1);

	vars.get<GBuffer>("gBuffer")->position->unbind(0);
	vars.get<GBuffer>("gBuffer")->normal->unbind(1);

	headTex->unbind(0);
	listTex->unbind(1);
	maxDepthTex->unbind(2);

	matrixBuffer->unbindBase(GL_UNIFORM_BUFFER, 0);

	//vars.get<Buffer>("xtmp")->unbindBase(GL_SHADER_STORAGE_BUFFER, 3);
	assert(glGetError() == GL_NO_ERROR);
}

uint32_t OFTS::GetNofWgsFill() const
{
	uint32_t wgSize = vars.getUint32(wgsizeParamName);
	glm::uvec2 const r = GetWindowSize();
	uint32_t const nofPixels = r.x * r.y;

	return (nofPixels / wgSize) + 1;
}

void OFTS::InitShadowMaskZBuffer()
{
	assert(glGetError() == GL_NO_ERROR);

	glm::uvec2 const lightRes = GetLightResolution();

	Framebuffer* fbo = vars.get<Framebuffer>(fboName);
	Program* program = vars.get<Program>(zbufferProgramName);
	Texture* maxDepthTex = vars.get<Texture>(maxDepthTexName);
	VertexArray* dummyVao = vars.get<VertexArray>(dummyVaoName);
	Buffer* matrixBuffer = vars.get<Buffer>(matrixBufferName);

	fbo->bind();
	glViewport(0, 0, lightRes.x, lightRes.y);
	glClear(GL_DEPTH_BUFFER_BIT);

	program->use();
	maxDepthTex->bind(0);

	dummyVao->bind();

	matrixBuffer->bindRange(GL_UNIFORM_BUFFER, 0, 12 * 16 * sizeof(float), 6 * sizeof(uint32_t));

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	matrixBuffer->unbindRange(GL_UNIFORM_BUFFER, 0);
	maxDepthTex->unbind(0);
	dummyVao->unbind();
	fbo->unbind();

	assert(glGetError() == GL_NO_ERROR);
}

void OFTS::CreateDummyVao()
{
	FUNCTION_PROLOGUE("ofts.objects", "renderModel");
	vars.reCreate<VertexArray>(dummyVaoName);
}

void OFTS::FillShadowMask()
{
	assert(glGetError() == GL_NO_ERROR);

	glm::uvec2 const screenRes = GetWindowSize();
	glm::vec3 const lightPos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));
	glm::uvec2 const lightRes = GetLightResolution();

	Framebuffer* fbo = vars.get<Framebuffer>(fboName);

	fbo->bind();
	glViewport(0, 0, lightRes.x, lightRes.y);

	Texture* headTex = vars.get<Texture>(headTexName);
	Texture* listTex = vars.get<Texture>(listTexName);
	std::shared_ptr<Texture> posTex = vars.get<GBuffer>("gBuffer")->position;
	Texture* shadowMask = vars.get<Texture>("shadowMask");

	Buffer* matrixBuffer = vars.get<Buffer>(matrixBufferName);

	VertexArray* vao = vars.get<VertexArray>(shadowMaskVaoName);

	Program* prog = vars.get<Program>(shadowMaskProgrammName);
	prog->set3fv("lightPos", glm::value_ptr(lightPos));
	prog->set1f("bias", vars.getFloat(biasParamName));
	prog->set2uiv("screenResolution", glm::value_ptr(screenRes));
	prog->setMatrix4fv("lightV", glm::value_ptr(lightV[0]), 6);
	prog->use();

	headTex->bindImage(0, 0, headTexFormat, GL_READ_WRITE, GL_TRUE, 0);
	listTex->bindImage(1);
	shadowMask->bindImage(2);

	posTex->bind(0);

	matrixBuffer->bindBase(GL_UNIFORM_BUFFER, 0);

	vao->bind();

	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glDrawArrays(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices);

	headTex->unbind(0);
	listTex->unbind(1);
	shadowMask->unbind(2);

	posTex->unbind(0);

	matrixBuffer->unbindBase(GL_UNIFORM_BUFFER, 0);

	vao->unbind();
	fbo->unbind();

	glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glViewport(0, 0, screenRes.x, screenRes.y);

	assert(glGetError() == GL_NO_ERROR);
}

void OFTS::CreateShadowMaskVao()
{
	FUNCTION_PROLOGUE("ofts.objects", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>(shadowMaskVaoName);
	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void OFTS::CreateShadowMaskFbo()
{
	FUNCTION_PROLOGUE("ofts.objects", "shadowMask");

	assert(glGetError() == GL_NO_ERROR);
	Framebuffer* fbo = vars.reCreate<Framebuffer>(fboName);

	glm::uvec2 const res = GetLightResolution();
	Texture* depthTex = vars.reCreate<Texture>(depthTexName, GL_TEXTURE_2D_ARRAY, GL_DEPTH_COMPONENT24, 1, res.x, res.y, 12);
	assert(glGetError() == GL_NO_ERROR);
	fbo->attachTexture(GL_DEPTH_ATTACHMENT, vars.get<Texture>(depthTexName));

	assert(glGetError() == GL_NO_ERROR);
}

glm::uvec2 OFTS::GetWindowSize() const
{
	return *vars.get<glm::uvec2>("windowSize");
}

glm::uvec2 OFTS::GetLightResolution() const
{
	uint32_t const res = vars.getUint32(resolutionParamName);
	return glm::uvec2(res, res);
}

glm::uvec2 OFTS::GetHeatmapResolution() const
{
	glm::uvec2 res;
	res.x = res.y = vars.getUint32(heatmapResParamName);

	return res;
}

glm::vec3 OFTS::GetLightPosition() const
{
	return glm::vec3(*vars.get<glm::vec4>("lightPosition"));
}

void OFTS::create(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	if (!IsValid())
	{
		return;
	}
	
	CreateDummyVao();
	CreateShadowMaskVao();
	CreateShadowMaskFbo();
	CompileShaders();
	CreateTextures();
	CreateBuffers();
	ComputeLightFrusta();
	CreateLightMatrices();
	ClearShadowMask();
	ClearTextures();

	uint8_t const mask = GetUsedFrustumMasks(viewMatrix);
	//PrintStats(mask);

	ifExistStamp("");
	
	ComputeHeatMap(mask);
	
	ifExistStamp("heatMap");

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	ComputeViewProjectionMatrices(mask);
	
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	ifExistStamp("matrices");

	ComputeIzb();
	
	ifExistStamp("izb");

	InitShadowMaskZBuffer();
	
	ifExistStamp("zbuffer");

	FillShadowMask();
	
	ifExistStamp("traversal");
}