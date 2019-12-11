#include <ShadowMapping.h>
#include <ShadowMappingShaders.h>

#include <Vars/Resource.h>
#include <FunctionPrologue.h>
#include <Model.h>
#include <Deferred.h>

using namespace ge::gl;

ShadowMapping::ShadowMapping(vars::Vars& vars) : ShadowMethod(vars)
{
	createPrograms();
	createShadowMaskVao();
}

ShadowMapping::~ShadowMapping()
{
	vars.erase("sm.objects");
}

void ShadowMapping::create(
	glm::vec4 const&,
	glm::mat4 const&,
	glm::mat4 const&) 
{
	createShadowMap();
	createFbo();
	createVao();
	createShadowMaskFbo();
	createLightViewMatrix();
	createLightProjMatrix();

	glm::mat4 const lightVP = _lightProjMatrix * _lightViewMatrix;

	renderShadowMap(lightVP);

	renderShadowMask(lightVP);
}

void ShadowMapping::createShadowMap()
{
	FUNCTION_PROLOGUE("sm.objects", "sm.args.resolution");

	unsigned int resolution = vars.getUint32("sm.args.resolution");

	Texture* sm = vars.reCreate<Texture>("sm.objects.shadowMap", GL_TEXTURE_2D, GL_DEPTH_COMPONENT24, 1, resolution, resolution);

	sm->texParameteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	sm->texParameteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	sm->texParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	sm->texParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	sm->texParameteri(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	sm->texParameteri(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);

	_texelSize = glm::vec2(1.f / resolution);
}

void ShadowMapping::createFbo()
{
	FUNCTION_PROLOGUE("sm.objects", "sm.args.resolution");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("sm.objects.FBO");

	fbo->attachTexture(GL_DEPTH_ATTACHMENT, vars.get<Texture>("sm.objects.shadowMap"));
}

void ShadowMapping::createVao()
{
	FUNCTION_PROLOGUE("sm.objects", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>("sm.objects.VAO");
	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void ShadowMapping::createShadowMaskVao()
{
	vars.reCreate<VertexArray>("sm.objects.shadowMaskVao");
}

void ShadowMapping::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("sm.objects", "shadowMask");

	Framebuffer* maskFbo = vars.reCreate<Framebuffer>("sm.objects.shadowMaskFbo");

	maskFbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	maskFbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void ShadowMapping::createPrograms()
{
	vars.reCreate<Program>("sm.objects.shadowMapProgram",
		std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, smVsSource));
	
	vars.reCreate<Program>("sm.objects.shadowMaskProgram",
		std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, smMaskVs),
		std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, smMaskFs));
}

void ShadowMapping::createLightViewMatrix()
{
	FUNCTION_PROLOGUE("sm.objects", "lightPosition", "lightUp", "lightView");

	glm::vec3 const pos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));
	glm::vec3 const* up = vars.get<glm::vec3>("lightUp");
	glm::vec3 const* dir = vars.get<glm::vec3>("lightView");

	_lightViewMatrix = glm::lookAt(pos, pos + *dir, *up);
}

void ShadowMapping::createLightProjMatrix()
{
	FUNCTION_PROLOGUE("sm.objects", "lightPosition", "sm.args.near", "sm.args.far", "sm.args.fovy", "sm.args.resolution");

	float nearZ = vars.getFloat("sm.args.near");
	float farZ = vars.getFloat("sm.args.far");
	float fovY = vars.getFloat("sm.args.fovy");

	_lightProjMatrix = glm::perspective(fovY, 1.f, nearZ, farZ);
}

void ShadowMapping::renderShadowMap(glm::mat4 const& lightVP)
{
	unsigned int const resolution = vars.getUint32("sm.args.resolution");

	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(5, 10);
	glViewport(0, 0, resolution, resolution);
	glEnable(GL_DEPTH_TEST);

	vars.get<Framebuffer>("sm.objects.FBO")->bind();
	glClear(GL_DEPTH_BUFFER_BIT);
	vars.get<VertexArray>("sm.objects.VAO")->bind();

	Program* program = vars.get<Program>("sm.objects.shadowMapProgram");

	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));
	program->use();

	glDrawArrays(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices);

	vars.get<VertexArray>("sm.objects.VAO")->unbind();
	vars.get<Framebuffer>("sm.objects.FBO")->unbind();

	glDisable(GL_POLYGON_OFFSET_FILL);
}

void ShadowMapping::renderShadowMask(glm::mat4 const& lightVP)
{
	glm::uvec2 windowSize = *vars.get<glm::uvec2>("windowSize");
	glViewport(0, 0, windowSize.x, windowSize.y);

	vars.get<Framebuffer>("sm.objects.shadowMaskFbo")->bind();
	vars.get<VertexArray>("sm.objects.shadowMaskVao")->bind();

	Program* program = vars.get<Program>("sm.objects.shadowMaskProgram");
	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));
	program->set1i("pcfSize", vars.getUint32("sm.args.pcf"));
	program->set2f("texelSize", _texelSize.x, _texelSize.y);
	program->use();

	vars.get<GBuffer>("gBuffer")->position->bind(0);
	vars.get<Texture>("sm.objects.shadowMap")->bind(1);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<VertexArray>("sm.objects.shadowMaskVao")->unbind();
	vars.get<Framebuffer>("sm.objects.shadowMaskFbo")->unbind();
}
