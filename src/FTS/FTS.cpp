#include <FTS.h>
#include <FTS_shaderGen.h>

#include <geGL/Texture.h>
#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/Framebuffer.h>
#include <geGL/VertexArray.h>

#include <ifExistStamp.h>
#include <FunctionPrologue.h>
#include <Model.h>
#include <Deferred.h>

using namespace ge;
using namespace gl;


constexpr const char* wgsizeParamName = "fts.args.wgSize";
constexpr const char* resolutionParamName = "fts.args.resolution";
constexpr const char* nearParamName = "fts.args.nearZ";
constexpr const char* farParamName = "fts.args.farZ";
constexpr const char* fovyParamName = "fts.args.fovY";

constexpr const char* listTexName  = "fts.objects.listTex";
constexpr const char* headTexName  = "fts.objects.headTex";
constexpr const char* mutexTexName = "fts.objects.mutexTex";

constexpr const char* samplerName = "fts.objects.posSampler";

constexpr const char* fillProgramName = "fts.objects.fillProgram";

constexpr const GLuint listTexFormat =  GL_R32I;
constexpr const GLuint headTexFormat =  GL_R32I;
constexpr const GLuint mutexTexFormat = GL_R32UI;


FTS::FTS(vars::Vars& vars) : ShadowMethod(vars)
{
	IsValid = IsConservativeRasterizationSupported();

	CreateShadowMaskVao();
	CompileShaders();
	CreateSampler();
}

FTS::~FTS()
{
	vars.erase("fts.objects");
}

void FTS::CreateShadowMaskVao()
{
	if (!IsValid) return;

	vars.reCreate<VertexArray>("fts.objects.shadowMaskVao");
}

void FTS::CreateTextures()
{
	CreateHeadTex();
	CreateLinkedListTex();
	CreateMutexTex();
}

void FTS::CreateSampler()
{
	Sampler* sampler = vars.reCreate<Sampler>(samplerName);
	sampler->setMinFilter(GL_NEAREST);
	sampler->setMagFilter(GL_NEAREST);
}

void FTS::CreateHeadTex()
{
	FUNCTION_PROLOGUE("fts.objects", resolutionParamName, "renderModel");

	uint32_t const res = vars.getUint32(resolutionParamName);
	CreateTexture2D(headTexName, headTexFormat, res, res);
}

void FTS::CreateLinkedListTex()
{
	FUNCTION_PROLOGUE("fts.objects", "windowSize", "renderModel");

	glm::uvec2 const windowSize = *vars.get<glm::uvec2>("windowSize");
	CreateTexture2D(listTexName, listTexFormat, windowSize.x, windowSize.y);
}

void FTS::CreateMutexTex()
{
	FUNCTION_PROLOGUE("fts.objects", resolutionParamName, "renderModel");

	uint32_t const res = vars.getUint32(resolutionParamName);
	CreateTexture2D(mutexTexName, mutexTexFormat, res, res);
}

void FTS::ClearTextures()
{
	int const clearVal = -1;
	unsigned int const mutexClearVal = 0;
	vars.get<Texture>(listTexName)->clear( 0, GL_RED_INTEGER, GL_INT, &clearVal);
	vars.get<Texture>(headTexName)->clear( 0, GL_RED_INTEGER, GL_INT, &clearVal);
	vars.get<Texture>(mutexTexName)->clear(0, GL_RED_INTEGER, GL_UNSIGNED_INT, &mutexClearVal);
}

void FTS::CreateTexture2D(char const* name, uint32_t format, uint32_t resX, uint32_t resY)
{
	auto t = vars.reCreate<Texture>(name, GL_TEXTURE_2D, format, 1, resX, resY);
}

void FTS::CompileShaders()
{
	FUNCTION_PROLOGUE("fts.objects", wgsizeParamName);
	FtsShaderGen gen;

	uint32_t const wgSize = vars.getUint32(wgsizeParamName);
	vars.reCreate<Program>(fillProgramName, gen.GetZbufferFillCS(wgSize));
}

void FTS::CreateIzb(glm::mat4 const& vp)
{
	glGetError();

	Program* program = vars.get<Program>(fillProgramName);
	
	Texture* headTex  = vars.get<Texture>(headTexName);
	Texture* listTex  = vars.get<Texture>(listTexName);
	Texture* mutexTex = vars.get<Texture>(mutexTexName);

	assert(program != nullptr);

	program->use();

	glm::mat4 const lightVP = CreateLightProjMatrix() * CreateLightViewMatrix();
	glm::uvec2 const screenRes = *vars.get<glm::uvec2>("windowSize");

	uint32_t const res = vars.getUint32(resolutionParamName);
	glm::uvec2 const lightRes = glm::uvec2(res, res);

	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));
	program->set2uiv("screenResolution", glm::value_ptr(screenRes));
	program->set2uiv("lightResolution", glm::value_ptr(lightRes));

	headTex->bindImage(0, 0, headTexFormat, GL_READ_WRITE);
	listTex->bindImage(1, 0, listTexFormat, GL_READ_WRITE);
	mutexTex->bindImage(2, 0, mutexTexFormat, GL_READ_WRITE);

	vars.get<GBuffer>("gBuffer")->position->bind(0);
	vars.get<Sampler>(samplerName)->bind(0);

	uint32_t const nofWgs = GetNofWgsFill();

	glDispatchCompute(nofWgs, 1, 1);

	glFinish();

	headTex->unbind(0);
	listTex->unbind(1);
	mutexTex->unbind(2);
	assert(glGetError() == GL_NO_ERROR);
}

glm::mat4 FTS::CreateLightViewMatrix() const
{
	glm::vec3 const pos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));
	glm::vec3 const* up = vars.get<glm::vec3>("lightUp");
	glm::vec3 const* dir = vars.get<glm::vec3>("lightView");

	return glm::lookAt(pos, pos + *dir, *up);
}

glm::mat4 FTS::CreateLightProjMatrix() const
{
	float nearZ = vars.getFloat(nearParamName);
	float farZ = vars.getFloat(farParamName);
	float fovY = vars.getFloat(fovyParamName);

	return glm::perspective(fovY, 1.f, nearZ, farZ);
}

uint32_t FTS::GetNofWgsFill() const
{
	uint32_t wgSize = vars.getUint32(wgsizeParamName);
	glm::uvec2 const r = *vars.get<glm::uvec2>("windowSize");
	uint32_t const nofPixels = r.x * r.y;

	return (nofPixels / wgSize) + 1;
}

void FTS::create(glm::vec4 const& lightPosition,
	glm::mat4 const& viewMatrix,
	glm::mat4 const& projectionMatrix)
{
	CreateTextures();
	CompileShaders();

	ClearTextures();
	CreateIzb(projectionMatrix * viewMatrix);
}