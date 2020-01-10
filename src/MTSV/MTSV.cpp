#include "MTSV.h"
#include "MTSV_shaders.h"

#include <geGL/Texture.h>
#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/Framebuffer.h>
#include <geGL/VertexArray.h>

#include <ifExistStamp.h>
#include <FunctionPrologue.h>
#include <Model.h>
#include <Deferred.h>

#include <fstream>

using namespace ge;
using namespace gl;

MTSV::MTSV(vars::Vars& vars) : ShadowMethod(vars)
{
	createShadowMaskProgram();
	createSupportBuffer();
	createShadowMaskVao();
}

MTSV::~MTSV()
{
	vars.erase("mtsv.objects");
}

void MTSV::create(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	getNofTriangles();
	createBuildProgram();
	createNodeBuffer();
	createTriangleBuffer();
	createShadowMaskFbo();

	ifExistStamp("");
	clearSupportBuffer();
	buildMetricTree(lightPosition);
	ifExistStamp("treeBuild");

	fillShadowMask(lightPosition);
	ifExistStamp("traverse");
}

void MTSV::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("mtsv.objects", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("mtsv.objects.FBO");
	fbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	fbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void MTSV::getNofTriangles()
{
	FUNCTION_PROLOGUE("mtsv.objects", "renderModel");

	size_t const nofVertices = vars.get<RenderModel>("renderModel")->nofVertices;
	NofTriangles = uint32_t(nofVertices) / 3;
}

void MTSV::createBuildProgram()
{
	FUNCTION_PROLOGUE("mtsv.objects", "mtsv.args.wgSize", "mtsv.args.useFrontFaceCulling");

	//std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\kkk.glsl");
	//std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());

	uint32_t const wgSize = vars.getUint32("mtsv.args.wgSize");

	std::string s = "#version 450 core\n" + (vars.getBool("mtsv.args.useFrontFaceCulling") ? Shader::define("USE_FF_CULLING") : "") + Shader::define("WG_SIZE", wgSize) + getMtsvBuildCsShader();

	vars.reCreate<Program>("mtsv.objects.buildProgram", std::make_shared<Shader>(GL_COMPUTE_SHADER, "#version 450 core\n", Shader::define("WG_SIZE", wgSize), getMtsvBuildCsShader()) );
	//vars.reCreate<Program>("mtsv.objects.buildProgram", std::make_shared<Shader>(GL_COMPUTE_SHADER, program));
}

void MTSV::createShadowMaskProgram()
{
	//std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\lll.glsl");
	//std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());

	vars.reCreate<Program>("mtsv.objects.shadowMaskProgram",
		std::make_shared<Shader>(GL_VERTEX_SHADER,   getMtsvShadowMaskVs()),
		std::make_shared<Shader>(GL_FRAGMENT_SHADER, getMtsvShadowMaskFs()) );
		//std::make_shared<Shader>(GL_FRAGMENT_SHADER, program));
}

void MTSV::createNodeBuffer()
{
	FUNCTION_PROLOGUE("mtsv.objects", "renderModel");

	size_t const nodeSize = 4 * sizeof(float) + 4 * sizeof(uint32_t); //Node struct on the GPU
	size_t const nofVertices = vars.get<RenderModel>("renderModel")->nofVertices;

	vars.reCreate<Buffer>("mtsv.objects.nodeBuffer", 4 * (nofVertices + 1) * nodeSize);
}

void MTSV::createTriangleBuffer()
{
	FUNCTION_PROLOGUE("mtsv.objects", "renderModel");

	vars.reCreate<Buffer>("mtsv.objects.triangleBuffer", NofTriangles * 3 * 4 * sizeof(float));
}

void MTSV::createSupportBuffer()
{
	vars.reCreate<Buffer>("mtsv.objects.supportBuffer", 4 * sizeof(uint32_t));
}

void MTSV::clearSupportBuffer()
{
	vars.get<Buffer>("mtsv.objects.supportBuffer")->clear(GL_R32UI, GL_RED, GL_INT);
}

void MTSV::createShadowMaskVao()
{
	vars.reCreate<VertexArray>("mtsv.objects.VAO");
}

void MTSV::buildMetricTree(glm::vec3 const& lightPos)
{
	Program* program = vars.get<Program>("mtsv.objects.buildProgram");
	program->use();
	program->set3fv("lightPos", glm::value_ptr(lightPos));
	program->set1f("delta", glm::half_pi<float>());
	program->set1ui("triangle_count", NofTriangles);

	vars.get<RenderModel>("renderModel")->vertices->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("mtsv.objects.triangleBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("mtsv.objects.nodeBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 2);
	vars.get<Buffer>("mtsv.objects.supportBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 3);

	//glDispatchCompute(vars.getUint32("mtsv.args.numWg"), 1, 1);

	{
		uint32_t const wgSize = vars.getUint32("mtsv.args.wgSize");
		uint32_t const nofWgs = NofTriangles / wgSize + 1;
		glDispatchCompute(nofWgs, 1, 1);
	}

	vars.get<RenderModel>("renderModel")->vertices->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("mtsv.objects.triangleBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("mtsv.objects.nodeBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 2);
	vars.get<Buffer>("mtsv.objects.supportBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 3);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void MTSV::setWindowViewport()
{
	glm::uvec2 const windowSize = *vars.get<glm::uvec2>("windowSize");
	glViewport(0, 0, windowSize.x, windowSize.y);
}

void MTSV::fillShadowMask(glm::vec3 const& lightPos)
{
	Program* program = vars.get<Program>("mtsv.objects.shadowMaskProgram");
	program->use();
	program->set3fv("lightPos", glm::value_ptr(lightPos));
	program->set1f("bias", vars.getFloat("mtsv.args.bias"));

	vars.get<VertexArray>("mtsv.objects.VAO")->bind();
	vars.get<Framebuffer>("mtsv.objects.FBO")->bind();

	setWindowViewport();

	vars.get<GBuffer>("gBuffer")->position->bind(0);

	vars.get<Buffer>("mtsv.objects.nodeBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("mtsv.objects.supportBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("mtsv.objects.triangleBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<Buffer>("mtsv.objects.nodeBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("mtsv.objects.supportBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("mtsv.objects.triangleBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 2);

	vars.get<VertexArray>("mtsv.objects.VAO")->unbind();
	vars.get<Framebuffer>("mtsv.objects.FBO")->unbind();
}

