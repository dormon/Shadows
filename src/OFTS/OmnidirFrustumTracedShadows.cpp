#include <OmnidirFrustumTracedShadows.h>
#include <FTS/FtsShaderGen.h>

#include <algorithm>
#include <cassert>

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

constexpr unsigned int NOF_SIDES = 6u;

OmnidirFrustumTracedShadows::OmnidirFrustumTracedShadows(vars::Vars& vars) : ShadowMethod(vars)
{
	IsValid = IsConservativeRasterizationSupported();

	createShadowMaskVao();
}

OmnidirFrustumTracedShadows::~OmnidirFrustumTracedShadows()
{
	vars.erase("ofts");
}

void OmnidirFrustumTracedShadows::create(glm::vec4 const&, glm::mat4 const&, glm::mat4 const&)
{
	if (!IsValid) return;

	updateConstants();
	createBuffers();
	createVao();
	createFbo();
	createShaders();
	createShadowMaskFbo();

	renderIzb();

	createShadowMask();
}

void OmnidirFrustumTracedShadows::createVao()
{
	FUNCTION_PROLOGUE("ofts", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>("ofts.VAO");

	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void OmnidirFrustumTracedShadows::createFbo()
{
	FUNCTION_PROLOGUE("ofts", "args.ofts.resolution");

	unsigned int const res = vars.getUint32("args.ofts.resolution");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("ofts.FBO");
	fbo->setDefaultWidth(res);
	fbo->setDefaultHeight(res);
	fbo->setDefaultLayers(NOF_SIDES);
}

void OmnidirFrustumTracedShadows::createBuffers()
{
	FUNCTION_PROLOGUE("ofts", "args.ofts.resolution", "args.ofts.depth", "renderModel");

	unsigned int const res = vars.getUint32("args.ofts.resolution");
	unsigned int const depth = vars.getUint32("args.ofts.depth");
	unsigned int const nofVertices = vars.get<RenderModel>("renderModel")->nofVertices;

	vars.reCreate<Buffer>("ofts.izb", NOF_SIDES * res * res * depth * sizeof(uint32_t));
	vars.reCreate<Buffer>("ofts.atomicCounter", NOF_SIDES * res * res * sizeof(uint32_t));
	vars.reCreate<Buffer>("ofts.frusta", (nofVertices / 3) * 4 * 4 * sizeof(float));
}

void OmnidirFrustumTracedShadows::createShaders()
{
	FUNCTION_PROLOGUE("ofts", "args.ofts.wgSize", "args.ofts.resolution", "args.ofts.depth");

	unsigned int const res = vars.getUint32("args.ofts.resolution");
	unsigned int const depth = vars.getUint32("args.ofts.depth");
	unsigned int const wgSize = vars.getUint32("args.ofts.wgSize");
	FtsShaderGen shaderGen;

	glm::uvec3 r = glm::uvec3(res, res, depth);

	vars.reCreate<Program>("ofts.fillProgram", shaderGen.GetIzbFillProgramOmnidirShaders(r));
	vars.reCreate<Program>("ofts.shadowMaskRaytrace", shaderGen.GetIzbTraversalProgramOmnidirRaytraceShaders(r));
	vars.reCreate<Program>("ofts.shadowMaskFrusta", shaderGen.GetIzbTraversalProgramOmnidirFrustaShaders(r));
	vars.reCreate<Program>("ofts.frustaPreprocessCS", shaderGen.GetTrianglePreprocessCSShader(wgSize));
}

void OmnidirFrustumTracedShadows::createShadowMaskVao()
{
	if (!IsValid) return;

	vars.reCreate<VertexArray>("ofts.shadowMaskVao");
}

void OmnidirFrustumTracedShadows::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("ofts", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("ofts.shadowMaskFbo");
	fbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	fbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void OmnidirFrustumTracedShadows::updateConstants()
{
	FUNCTION_PROLOGUE("ofts", "renderModel", "args.ofts.wgSize");

	unsigned int const wgSize = vars.getUint32("args.ofts.wgSize");
	unsigned int const nofVertices = vars.get<RenderModel>("renderModel")->nofVertices;

	vars.reCreate<uint32_t>("ofts.nofWgs", (nofVertices / 3) / wgSize + 1);
}

void OmnidirFrustumTracedShadows::renderIzb()
{
	assert(glGetError() == GL_NO_ERROR);

	Buffer* atomicCounter = vars.get<Buffer>("ofts.atomicCounter");
	Program* program = vars.get<Program>("ofts.fillProgram");
	unsigned int const res = vars.getUint32("args.ofts.resolution");
	float const nearZ = vars.getFloat("args.ofts.near");
	float const farZ = vars.getFloat("args.ofts.far");
	glm::vec4 lightPosition = *vars.get<glm::vec4>("lightPosition");

	atomicCounter->clear(GL_R32UI, GL_RED, GL_INT);

	vars.get<Framebuffer>("ofts.FBO")->bind();
	vars.get<VertexArray>("ofts.VAO")->bind();

	glViewport(0, 0, res, res);
	glDisable(GL_DEPTH_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);

	program->use();
	program->set4fv("lightPos", glm::value_ptr(lightPosition));
	program->set1f("far", farZ);
	program->set1f("near", nearZ);

	vars.get<Buffer>("ofts.izb")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	atomicCounter->bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDrawArraysInstanced(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices, NOF_SIDES);

	vars.get<Framebuffer>("ofts.FBO")->unbind();
	vars.get<VertexArray>("ofts.VAO")->unbind();

	vars.get<Buffer>("ofts.izb")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	atomicCounter->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
}

void OmnidirFrustumTracedShadows::createShadowMask()
{
	glm::uvec2 windowSize = *vars.get<glm::uvec2>("windowSize");

	bool const useFrusta = vars.getBool("args.ofts.useFrusta");
	Buffer* vertexOrFrustumBuffer = nullptr;
	Program* raytraceOrFrustumProgram = nullptr;

	if (useFrusta)
	{
		preprocessFrusta();
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		vertexOrFrustumBuffer = vars.get<Buffer>("ofts.frusta");
		raytraceOrFrustumProgram = vars.get<Program>("ofts.shadowMaskFrusta");
	}
	else
	{
		vertexOrFrustumBuffer = vars.get<RenderModel>("renderModel")->vertices.get();
		raytraceOrFrustumProgram = vars.get<Program>("ofts.shadowMaskRaytrace");
	}

	glEnable(GL_DEPTH_TEST);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glViewport(0, 0, windowSize.x, windowSize.y);

	vars.get<Framebuffer>("ofts.shadowMaskFbo")->bind();
	vars.get<VertexArray>("ofts.shadowMaskVao")->bind();

	raytraceOrFrustumProgram->use();
	raytraceOrFrustumProgram->set4fv("lightPos", glm::value_ptr(*vars.get<glm::vec4>("lightPosition")));

	if (!useFrusta)
	{
		raytraceOrFrustumProgram->set1f("bias", vars.getFloat("args.ofts.bias"));
	}

	vars.get<GBuffer>("gBuffer")->position->bind(0);

	vars.get<Buffer>("ofts.izb")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("ofts.atomicCounter")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vertexOrFrustumBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<Framebuffer>("ofts.shadowMaskFbo")->unbind();
	vars.get<VertexArray>("ofts.shadowMaskVao")->unbind();

	vars.get<Buffer>("ofts.izb")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("ofts.atomicCounter")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vertexOrFrustumBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, 2);
}

void OmnidirFrustumTracedShadows::preprocessFrusta()
{
	Program* program = vars.get<Program>("ofts.frustaPreprocessCS");

	program->use();
	program->set4fv("lightPos", glm::value_ptr(*vars.get<glm::vec4>("lightPosition")));
	program->set1ui("nofTriangles", vars.get<RenderModel>("renderModel")->nofVertices / 3);

	vars.get<RenderModel>("renderModel")->vertices->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("ofts.frusta")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDispatchCompute(*vars.get<uint32_t>("ofts.nofWgs"), 1, 1);

	vars.get<RenderModel>("renderModel")->vertices->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("ofts.frusta")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
}

bool OmnidirFrustumTracedShadows::IsConservativeRasterizationSupported() const
{
	int NumberOfExtensions;
	glGetIntegerv(GL_NUM_EXTENSIONS, &NumberOfExtensions);
	for (int i = 0; i < NumberOfExtensions; i++)
	{
		const char* ccc = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));

		if (strcmp(ccc, "GL_NV_conservative_raster") == 0)
		{
			return true;
		}
	}

	return false;
}