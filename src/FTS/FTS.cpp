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
constexpr const char* counterTexName = "fts.objects.counterTex";
constexpr const char* maxDepthTexName = "fts.objects.maxDepthTex";

constexpr const char* pointSamplerName = "fts.objects.pointSampler";

constexpr const char* fillProgramName = "fts.objects.fillProgram";
constexpr const char* zbufferProgramName = "fts.objects.zbufferProgram";
constexpr const char* shadowMaskProgrammName = "fts.objects.shadowMaskProgram";

constexpr const char* fboName = "fts.objects.fboName";
constexpr const char* depthTexName = "fts.objects.depthTex";
constexpr const char* shadowMaskVaoName = "fts.objects.shadowMaskVao";
constexpr const char* dummyVaoName = "fts.objects.dummyVao";

constexpr const GLuint listTexFormat =  GL_R32I;
constexpr const GLuint headTexFormat =  GL_R32I;
constexpr const GLuint counterTexFormat = GL_R32UI;
constexpr const GLuint maxDepthTexFormat = GL_R32F;

FTS::FTS(vars::Vars& vars) : ShadowMethod(vars)
{
	IsValid = IsConservativeRasterizationSupported();

	CreateSampler();
	CreateDummyVao();
}

FTS::~FTS()
{
	vars.erase("fts.objects");
}

void FTS::CreateTextures()
{
	CreateHeadTex();
	CreateLinkedListTex();
	CreateCounterTex();
	CreateMaxDepthTex();
}

void FTS::CreateSampler()
{
	Sampler* sampler = vars.reCreate<Sampler>(pointSamplerName);
	sampler->setMinFilter(GL_NEAREST);
	sampler->setMagFilter(GL_NEAREST);
}

void FTS::CreateHeadTex()
{
	FUNCTION_PROLOGUE("fts.objects", resolutionParamName, "renderModel");

	glm::uvec2 const lightRes = GetLightResolution();
	CreateTexture2D(headTexName, headTexFormat, lightRes.x, lightRes.y);
}

void FTS::CreateLinkedListTex()
{
	FUNCTION_PROLOGUE("fts.objects", "windowSize", "renderModel");

	glm::uvec2 const windowSize = GetWindowSize();
	CreateTexture2D(listTexName, listTexFormat, windowSize.x, windowSize.y);
}

void FTS::CreateCounterTex()
{
	FUNCTION_PROLOGUE("fts.objects", resolutionParamName, "renderModel");

	glm::uvec2 const lightRes = GetLightResolution();
	CreateTexture2D(counterTexName, counterTexFormat, lightRes.x, lightRes.y);
}

void FTS::CreateMaxDepthTex()
{
	FUNCTION_PROLOGUE("fts.objects", resolutionParamName, "renderModel");

	glm::uvec2 const lightRes = GetLightResolution();
	CreateTexture2D(maxDepthTexName, maxDepthTexFormat, lightRes.x, lightRes.y);
}

void FTS::ClearTextures()
{
	assert(glGetError() == GL_NO_ERROR);

	int const clearVal = -1;
	vars.get<Texture>(listTexName)->clear( 0, GL_RED_INTEGER, GL_INT, &clearVal);
	vars.get<Texture>(headTexName)->clear( 0, GL_RED_INTEGER, GL_INT, &clearVal);

	float const val = 1.f;
	vars.get<Texture>("shadowMask")->clear(0, GL_RED, GL_FLOAT, &val);

	vars.get<Texture>(counterTexName)->clear(0, GL_RED_INTEGER, GL_INT);
	vars.get<Texture>(maxDepthTexName)->clear(0, GL_RED, GL_FLOAT);

	assert(glGetError() == GL_NO_ERROR);
}

void FTS::CreateTexture2D(char const* name, uint32_t format, uint32_t resX, uint32_t resY)
{
	auto t = vars.reCreate<Texture>(name, GL_TEXTURE_2D, format, 1, resX, resY);
}

void FTS::CompileShaders()
{
	CreateIzbFillProgram();
	CreateZbufferFillProgram();
	CreateShadowMaskProgram();
}

void FTS::CreateIzbFillProgram()
{
	FUNCTION_PROLOGUE("fts.objects", wgsizeParamName);
	FtsShaderGen gen;

	uint32_t const wgSize = vars.getUint32(wgsizeParamName);
	vars.reCreate<Program>(fillProgramName, gen.GetIzbFillCS(wgSize));
}

void FTS::CreateZbufferFillProgram()
{
	FUNCTION_PROLOGUE("fts.objects");

	FtsShaderGen gen;
	vars.reCreate<Program>(zbufferProgramName, gen.GetZBufferFillVS(), gen.GetZBufferFillFS());
}

void FTS::CreateShadowMaskProgram()
{
	FUNCTION_PROLOGUE("fts.objects");

	FtsShaderGen gen;
	vars.reCreate<Program>(shadowMaskProgrammName, gen.GetShadowMaskVS(), gen.GetShadowMaskGS(), gen.GetShadowMaskFS());
}

void FTS::CreateIzb(glm::mat4 const& vp, glm::mat4 const& lightVP)
{
	assert(glGetError() == GL_NO_ERROR);

	Program* program = vars.get<Program>(fillProgramName);

	Texture* headTex  = vars.get<Texture>(headTexName);
	Texture* listTex  = vars.get<Texture>(listTexName);
	Texture* counterTex = vars.get<Texture>(counterTexName);
	Texture* maxDepthTex = vars.get<Texture>(maxDepthTexName);

	assert(program != nullptr);

	program->use();

	glm::uvec2 const screenRes = GetWindowSize();
	glm::uvec2 const lightRes = GetLightResolution();
	glm::vec3 const lightPos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));

	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));
	program->set2uiv("screenResolution", glm::value_ptr(screenRes));
	program->set2uiv("lightResolution", glm::value_ptr(lightRes));
	program->set3fv("lightPos", glm::value_ptr(lightPos));

	headTex->bindImage(0, 0, headTexFormat, GL_READ_WRITE);
	listTex->bindImage(1, 0, listTexFormat, GL_READ_WRITE);
	counterTex->bindImage(2, 0, counterTexFormat, GL_WRITE_ONLY);
	maxDepthTex->bindImage(3, 0, maxDepthTexFormat, GL_READ_WRITE);

	vars.get<GBuffer>("gBuffer")->position->bind(0);
	vars.get<GBuffer>("gBuffer")->normal->bind(1);

	uint32_t const nofWgs = GetNofWgsFill();

	glDispatchCompute(nofWgs, 1, 1);

	vars.get<GBuffer>("gBuffer")->position->unbind(0);
	vars.get<GBuffer>("gBuffer")->normal->unbind(1);

	headTex->unbind(0);
	listTex->unbind(1);
	counterTex->unbind(2);
	maxDepthTex->unbind(3);

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
	glm::uvec2 const r = GetWindowSize();
	uint32_t const nofPixels = r.x * r.y;

	return (nofPixels / wgSize) + 1;
}

//-----------------Early Z optimization----------

void FTS::CreateDummyVao()
{
	vars.reCreate<VertexArray>(dummyVaoName);
}

void FTS::InitShadowMaskZBuffer()
{
	assert(glGetError() == GL_NO_ERROR);
	
	glm::uvec2 const lightRes = GetLightResolution();

	Framebuffer* fbo = vars.get<Framebuffer>(fboName);
	Program* program = vars.get<Program>(zbufferProgramName);
	Texture* maxDepthTex = vars.get<Texture>(maxDepthTexName);
	VertexArray* dummyVao = vars.get<VertexArray>(dummyVaoName);

	fbo->bind();
	glViewport(0, 0, lightRes.x, lightRes.y);
	glClear(GL_DEPTH_BUFFER_BIT);

	program->use();
	maxDepthTex->bind(0);

	dummyVao->bind();

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	
	maxDepthTex->unbind(0);
	dummyVao->unbind();
	fbo->unbind();

	assert(glGetError() == GL_NO_ERROR);
}


//-----------------Shadow Mask-------------------

void FTS::CreateShadowMaskVao()
{
	FUNCTION_PROLOGUE("fts.objects", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>(shadowMaskVaoName);
	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void FTS::CreateShadowMaskFbo()
{
	FUNCTION_PROLOGUE("fts.objects", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>(fboName);

	glm::uvec2 const res = GetLightResolution();
	Texture* depthTex = vars.reCreate<Texture>(depthTexName, GL_TEXTURE_RECTANGLE, GL_DEPTH24_STENCIL8, 1, res.x, res.y);

	fbo->attachTexture(GL_DEPTH_ATTACHMENT, vars.get<Texture>(depthTexName));
}

void FTS::CreateShadowMask(glm::mat4 const& lightVP)
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

	VertexArray* vao = vars.get<VertexArray>(shadowMaskVaoName);

	Program* prog = vars.get<Program>(shadowMaskProgrammName);
	prog->setMatrix4fv("lightVP", glm::value_ptr(lightVP));
	prog->set3fv("lightPos", glm::value_ptr(lightPos));
	prog->set1f("bias", vars.getFloat("fts.args.traversalBias"));
	prog->set2uiv("screenResolution", glm::value_ptr(screenRes));
	prog->use();

	headTex->bindImage(0);
	listTex->bindImage(1);
	shadowMask->bindImage(2);

	posTex->bind(0);
	
	vao->bind();

	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glDrawArrays(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices);

	headTex->unbind(0);
	listTex->unbind(1);
	shadowMask->unbind(2);

	posTex->unbind(0);

	vao->unbind();
	fbo->unbind();

	glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glViewport(0, 0, screenRes.x, screenRes.y);

	assert(glGetError() == GL_NO_ERROR);
}

glm::uvec2 FTS::GetWindowSize() const
{
	return *vars.get<glm::uvec2>("windowSize");
}

glm::uvec2 FTS::GetLightResolution() const
{
	uint32_t const res = vars.getUint32(resolutionParamName);
	return glm::uvec2(res, res);
}

void FTS::create(glm::vec4 const& lightPosition,
	glm::mat4 const& viewMatrix,
	glm::mat4 const& projectionMatrix)
{
	if(!IsValid)
	{
		return;
	}

	ifExistStamp("");

	CreateShadowMaskVao();
	CreateShadowMaskFbo();
	CompileShaders();
	CreateTextures();

	ClearTextures();

	glm::mat4 const lightVP = CreateLightProjMatrix() * CreateLightViewMatrix();
	glm::mat4 const vp = projectionMatrix * viewMatrix;

	//ifExistStamp("ftsSetup");

	CreateIzb(vp, lightVP);

	//ifExistStamp("ftsCreate");

	InitShadowMaskZBuffer();

	//ifExistStamp("ftsZFill");

	CreateShadowMask(lightVP);

	//ifExistStamp("ftsTraverse");
	ifExistStamp("fts");
}