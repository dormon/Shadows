#include "DPSM.h"
#include "DPSM_shaders.h"

#include <geGL/Texture.h>
#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/Framebuffer.h>
#include <geGL/VertexArray.h>

#include <Model.h>
#include <Deferred.h>

#include <FunctionPrologue.h>

#include <Vars/Vars.h>

using namespace ge;
using namespace gl;

constexpr const char* g_ShadowMapProgram = "dpsm.objects.shadowMapProgram";
constexpr const char* g_ShadowMapVAO = "dpsm.objects.shadowMapVAO";
constexpr const char* g_ShadowMapFBO = "dpsm.objects.shadowMapFBO";
constexpr const char* g_ShadowMapTexture = "dpsm.objects.shadowMapTexture";

constexpr const char* g_ShadowMaskProgram = "dpsm.objects.shadowMaskProgram";
constexpr const char* g_ShadowMaskVAO = "dpsm.objects.shadowMaskVAO";
constexpr const char* g_ShadowMaskFBO = "dpsm.objects.shadowMaskFBO";

constexpr const char* g_NearParam = "dpsm.args.near";
constexpr const char* g_FarParam = "dpsm.args.far";
constexpr const char* g_ResolutionParam = "dpsm.args.resolution";

DPSM::DPSM(vars::Vars& vars) : ShadowMethod(vars)
{
	createPrograms();
	createShadowMaskVao();
}

DPSM::~DPSM()
{
	vars.erase("dpsm.objects");
}

void DPSM::create(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	createShadowMap();
	createShadowMapFbo();
	createShadowMapVao();
	createShadowMaskFbo();

	createLightViewMatrix();

	renderShadowMap();
	renderShadowMask();
}

void DPSM::createShadowMap()
{
	FUNCTION_PROLOGUE("dpsm.objects", g_ResolutionParam);

	uint32_t const resolution = vars.getUint32(g_ResolutionParam);
	Texture* sm = vars.reCreate<Texture>(g_ShadowMapTexture, GL_TEXTURE_2D_ARRAY, GL_DEPTH_COMPONENT24, 1, resolution, resolution, 2);
	sm->texParameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	sm->texParameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	sm->texParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	sm->texParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	sm->texParameteri(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	sm->texParameteri(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
}

void DPSM::createShadowMapFbo()
{
	FUNCTION_PROLOGUE("dpsm.objects", g_ResolutionParam);

	Framebuffer* fbo = vars.reCreate<Framebuffer>(g_ShadowMapFBO);
	fbo->attachTexture(GL_DEPTH_ATTACHMENT, vars.get<Texture>(g_ShadowMapTexture));
}

void DPSM::createShadowMapVao()
{
	FUNCTION_PROLOGUE("dpsm.objects", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>(g_ShadowMapVAO);
	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void DPSM::createShadowMaskVao()
{
	vars.reCreate<VertexArray>(g_ShadowMaskVAO);
}

void DPSM::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("dpsm.objects", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>(g_ShadowMaskFBO);
	fbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	fbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void DPSM::createPrograms()
{
	createDpsmFillProgram();
	createDpsmShadowProgram();
}

void DPSM::createDpsmFillProgram()
{
	vars.reCreate<Program>(g_ShadowMapProgram,
		std::make_shared<Shader>(GL_VERTEX_SHADER, getDpsmCreateVS()),
		std::make_shared<Shader>(GL_GEOMETRY_SHADER, getDpsmCreateGS()),
		std::make_shared<Shader>(GL_FRAGMENT_SHADER, getDpsmCreateFS())
	);
}

void DPSM::createDpsmShadowProgram()
{
	vars.reCreate<Program>(g_ShadowMaskProgram,
		std::make_shared<Shader>(GL_VERTEX_SHADER, getDpsmFillVS()),
		std::make_shared<Shader>(GL_FRAGMENT_SHADER, getDpsmFillFS())
		);
}

void DPSM::createLightViewMatrix()
{
	glm::vec3 const pos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));
	glm::vec3 const up = glm::vec3(0.f, 1.f, 0.f);
	glm::vec3 const dir = glm::vec3(0, 0, 1);

	_lightViewMatrix = glm::lookAt(pos, pos - dir, up);
}

void DPSM::renderShadowMap()
{
	uint32_t const resolution = vars.getUint32(g_ResolutionParam);
	float const near = vars.getFloat(g_NearParam);
	float const far = vars.getFloat(g_FarParam);

	auto const fbo = vars.get<Framebuffer>(g_ShadowMapFBO);
	auto const vao = vars.get<VertexArray>(g_ShadowMapVAO);
	auto const program = vars.get<Program>(g_ShadowMapProgram);

	fbo->bind();
	vao->bind();

	glViewport(0, 0, resolution, resolution);
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	program
		->setMatrix4fv("lightV", glm::value_ptr(_lightViewMatrix))
		->set1f("nearClip", near)
		->set1f("farClip", far)
		->use();
	
	glDrawArrays(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices);

	vao->unbind();
	fbo->unbind();
}

void DPSM::renderShadowMask()
{
	float const near = vars.getFloat(g_NearParam);
	float const far = vars.getFloat(g_FarParam);
	glm::uvec2 const windowSize = *vars.get<glm::uvec2>("windowSize");

	auto const fbo = vars.get<Framebuffer>(g_ShadowMaskFBO);
	auto const vao = vars.get<VertexArray>(g_ShadowMaskVAO);
	auto const program = vars.get<Program>(g_ShadowMaskProgram);

	glViewport(0, 0, windowSize.x, windowSize.y);

	fbo->bind();
	program
		->setMatrix4fv("lightV", glm::value_ptr(_lightViewMatrix))
		->set1f("nearClip", near)
		->set1f("farClip", far)
		->use();
	vao->bind();
	vars.get<GBuffer>("gBuffer")->position->bind(0);
	vars.get<Texture>(g_ShadowMapTexture)->bind(1);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<GBuffer>("gBuffer")->position->unbind(0);
	vars.get<Texture>(g_ShadowMapTexture)->unbind(1);

	vao->unbind();
	fbo->unbind();
}
