#include <OmnidirDPM.h>
#include <DPM/DpmShaderGen.h>

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

OmnidirDpm::OmnidirDpm(vars::Vars& vars) : ShadowMethod(vars)
{
	IsValid = IsConservativeRasterizationSupported();

	createShadowMaskVao();
}

OmnidirDpm::~OmnidirDpm()
{
	vars.erase("odpm.objects");
}

void OmnidirDpm::create(glm::vec4 const&, glm::mat4 const&, glm::mat4 const&)
{
	if (!IsValid) return;

	updateConstants();
	createBuffers();
	createTriangleBuffer();
	createVao();
	createFbo();
	createShaders();
	createShadowMaskFbo();

	ifExistStamp("");
	renderIzb();
	ifExistStamp("IZB Create");
	createShadowMask();
	ifExistStamp("IZB Traversal");
}

void OmnidirDpm::createVao()
{
	FUNCTION_PROLOGUE("odpm.objects", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>("odpm.objects.VAO");

	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void OmnidirDpm::createFbo()
{
	FUNCTION_PROLOGUE("odpm.objects", "odpm.args.resolution");

	unsigned int const res = vars.getUint32("odpm.args.resolution");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("odpm.objects.FBO");
	fbo->setDefaultWidth(res);
	fbo->setDefaultHeight(res);
	fbo->setDefaultLayers(NOF_SIDES);
}

void OmnidirDpm::createBuffers()
{
	FUNCTION_PROLOGUE("odpm.objects", "odpm.args.resolution", "odpm.args.depth", "renderModel");

	unsigned int const res = vars.getUint32("odpm.args.resolution");
	unsigned int const depth = vars.getUint32("odpm.args.depth");
	unsigned int const nofVertices = vars.get<RenderModel>("renderModel")->nofVertices;

	vars.reCreate<Buffer>("odpm.objects.dpm", NOF_SIDES * res * res * depth * sizeof(uint32_t));
	vars.reCreate<Buffer>("odpm.objects.atomicCounter", NOF_SIDES * res * res * sizeof(uint32_t));
	vars.reCreate<Buffer>("odpm.objects.frusta", (nofVertices / 3) * 4 * 4 * sizeof(float));
}

void OmnidirDpm::createTriangleBuffer()
{
	FUNCTION_PROLOGUE("odpm.objects", "odpm.args.resolution", "odpm.args.depth", "renderModel");

	std::vector<float> const verts = vars.get<Model>("model")->getVertices();

	size_t const nofVerts = verts.size() / 3;

	std::vector<glm::vec4> v4;
	v4.reserve(nofVerts);

	for(size_t v = 0; v < nofVerts; ++v)
	{
		v4.push_back(glm::vec4(verts[3 * v + 0], verts[3 * v + 1], verts[3 * v + 2], 1));
	}

	vars.reCreate<Buffer>("odpm.objects.triangleBuffer", 4 * nofVerts * sizeof(float), v4.data());
}

void OmnidirDpm::createShaders()
{
	FUNCTION_PROLOGUE("odpm.objects", "odpm.args.wgSize", "odpm.args.resolution", "odpm.args.depth");

	unsigned int const res = vars.getUint32("odpm.args.resolution");
	unsigned int const depth = vars.getUint32("odpm.args.depth");
	unsigned int const wgSize = vars.getUint32("odpm.args.wgSize");
	DpmShaderGen shaderGen;

	glm::uvec3 r = glm::uvec3(res, res, depth);

	vars.reCreate<Program>("odpm.objects.fillProgram", shaderGen.GetDpmFillProgramOmnidirShaders(r));
	vars.reCreate<Program>("odpm.objects.shadowMaskRaytrace", shaderGen.GetDpmTraversalProgramOmnidirRaytraceShaders(r));
	vars.reCreate<Program>("odpm.objects.shadowMaskFrusta", shaderGen.GetDpmTraversalProgramOmnidirFrustaShaders(r));
	vars.reCreate<Program>("odpm.objects.frustaPreprocessCS", shaderGen.GetTrianglePreprocessCSShader(wgSize));
}

void OmnidirDpm::createShadowMaskVao()
{
	if (!IsValid) return;

	vars.reCreate<VertexArray>("odpm.objects.shadowMaskVao");
}

void OmnidirDpm::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("odpm.objects", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("odpm.objects.shadowMaskFbo");
	fbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	fbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void OmnidirDpm::updateConstants()
{
	FUNCTION_PROLOGUE("odpm.objects", "renderModel", "odpm.args.wgSize");

	unsigned int const wgSize = vars.getUint32("odpm.args.wgSize");
	unsigned int const nofVertices = vars.get<RenderModel>("renderModel")->nofVertices;

	vars.reCreate<uint32_t>("odpm.objects.nofWgs", (nofVertices / 3) / wgSize + 1);
}

void OmnidirDpm::renderIzb()
{
	assert(glGetError() == GL_NO_ERROR);

	Buffer* atomicCounter = vars.get<Buffer>("odpm.objects.atomicCounter");
	Program* program = vars.get<Program>("odpm.objects.fillProgram");
	unsigned int const res = vars.getUint32("odpm.args.resolution");
	float const nearZ = vars.getFloat("odpm.args.near");
	float const farZ = vars.getFloat("odpm.args.far");
	glm::vec4 lightPosition = *vars.get<glm::vec4>("lightPosition");

	atomicCounter->clear(GL_R32UI, GL_RED, GL_INT);

	vars.get<Framebuffer>("odpm.objects.FBO")->bind();
	vars.get<VertexArray>("odpm.objects.VAO")->bind();

	glViewport(0, 0, res, res);
	glDisable(GL_DEPTH_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);

	program->use();
	program->set4fv("lightPos", glm::value_ptr(lightPosition));
	program->set1f("far", farZ);
	program->set1f("near", nearZ);

	vars.get<Buffer>("odpm.objects.dpm")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	atomicCounter->bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDrawArraysInstanced(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices, NOF_SIDES);

	vars.get<Framebuffer>("odpm.objects.FBO")->unbind();
	vars.get<VertexArray>("odpm.objects.VAO")->unbind();

	vars.get<Buffer>("odpm.objects.dpm")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	atomicCounter->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
}

void OmnidirDpm::createShadowMask()
{
	glm::uvec2 windowSize = *vars.get<glm::uvec2>("windowSize");

	bool const useFrusta = vars.getBool("odpm.args.useFrusta");
	Buffer* vertexOrFrustumBuffer = nullptr;
	Program* raytraceOrFrustumProgram = nullptr;

	if (useFrusta)
	{
		preprocessFrusta();
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		vertexOrFrustumBuffer = vars.get<Buffer>("odpm.objects.frusta");
		raytraceOrFrustumProgram = vars.get<Program>("odpm.objects.shadowMaskFrusta");
	}
	else
	{
		vertexOrFrustumBuffer = vars.get<Buffer>("odpm.objects.triangleBuffer");
		raytraceOrFrustumProgram = vars.get<Program>("odpm.objects.shadowMaskRaytrace");
	}

	glEnable(GL_DEPTH_TEST);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glViewport(0, 0, windowSize.x, windowSize.y);

	vars.get<Framebuffer>("odpm.objects.shadowMaskFbo")->bind();
	vars.get<VertexArray>("odpm.objects.shadowMaskVao")->bind();

	raytraceOrFrustumProgram->use();
	raytraceOrFrustumProgram->set4fv("lightPos", glm::value_ptr(*vars.get<glm::vec4>("lightPosition")));

	if (!useFrusta)
	{
		raytraceOrFrustumProgram->set1f("bias", vars.getFloat("odpm.args.bias"));
	}

	vars.get<GBuffer>("gBuffer")->position->bind(0);

	vars.get<Buffer>("odpm.objects.dpm")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("odpm.objects.atomicCounter")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vertexOrFrustumBuffer->bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<Framebuffer>("odpm.objects.shadowMaskFbo")->unbind();
	vars.get<VertexArray>("odpm.objects.shadowMaskVao")->unbind();

	vars.get<Buffer>("odpm.objects.dpm")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("odpm.objects.atomicCounter")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vertexOrFrustumBuffer->unbindBase(GL_SHADER_STORAGE_BUFFER, 2);
}

void OmnidirDpm::preprocessFrusta()
{
	Program* program = vars.get<Program>("odpm.frustaPreprocessCS");

	program->use();
	program->set4fv("lightPos", glm::value_ptr(*vars.get<glm::vec4>("lightPosition")));
	program->set1ui("nofTriangles", vars.get<RenderModel>("renderModel")->nofVertices / 3);

	vars.get<RenderModel>("renderModel")->vertices->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("odpm.frusta")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDispatchCompute(*vars.get<uint32_t>("odpm.nofWgs"), 1, 1);

	vars.get<RenderModel>("renderModel")->vertices->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("odpm.frusta")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
}
