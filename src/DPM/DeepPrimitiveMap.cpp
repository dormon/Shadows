#include <DeepPrimitiveMap.h>
#include <DpmShaderGen.h>

#include <cassert>
#include <algorithm>

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

DeepPrimitiveMap::DeepPrimitiveMap(vars::Vars& vars) : ShadowMethod(vars)
{
	_isValid = IsConservativeRasterizationSupported();

	createShadowMaskVao();
}

DeepPrimitiveMap::~DeepPrimitiveMap()
{
	vars.erase("dpm.objects");
}

void DeepPrimitiveMap::create(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	if (!_isValid) return;

	createBuffers();
	createTriangleBuffer();
	createVao();
	createFbo();
	createShaders();
	createShadowMaskFbo();
	createLightViewMatrix();
	createLightProjMatrix();

	glm::mat4 const lightVP = _lightProjMatrix * _lightViewMatrix;

	ifExistStamp("");

	renderIzb(lightVP);

	ifExistStamp("createIZB");

	createShadowMask(lightVP);
	ifExistStamp("createShadowMask");
}

bool DeepPrimitiveMap::IsConservativeRasterizationSupported() const
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

void DeepPrimitiveMap::createBuffers()
{
	FUNCTION_PROLOGUE("dpm", "dpm.args.resolution", "dpm.args.depth");

	unsigned int res = vars.getUint32("dpm.args.resolution");
	unsigned int depth = vars.getUint32("dpm.args.depth");

	vars.reCreate<Buffer>("dpm.objects.dpm", res * res * depth * sizeof(uint32_t));
	vars.reCreate<Buffer>("dpm.objects.atomicCounter", res * res * sizeof(uint32_t));
}

void DeepPrimitiveMap::createTriangleBuffer()
{
	FUNCTION_PROLOGUE("dpm", "dpm.args.resolution", "dpm.args.depth", "renderModel");

	std::vector<float> const verts = vars.get<Model>("model")->getVertices();

	size_t const nofVerts = verts.size() / 3;

	std::vector<glm::vec4> v4;
	v4.reserve(nofVerts);

	for (size_t v = 0; v < nofVerts; ++v)
	{
		v4.push_back(glm::vec4(verts[3 * v + 0], verts[3 * v + 1], verts[3 * v + 2], 1));
	}

	vars.reCreate<Buffer>("dpm.objects.triangleBuffer", 4 * nofVerts * sizeof(float), v4.data());
}


void DeepPrimitiveMap::createVao()
{
	FUNCTION_PROLOGUE("dpm", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>("dpm.objects.VAO");

	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void DeepPrimitiveMap::createFbo()
{
	FUNCTION_PROLOGUE("dpm", "dpm.args.resolution");

	unsigned int const res = vars.getUint32("dpm.args.resolution");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("dpm.objects.FBO");
	fbo->setDefaultWidth(res);
	fbo->setDefaultHeight(res);
}

void DeepPrimitiveMap::createShaders()
{
	FUNCTION_PROLOGUE("dpm", "dpm.args.resolution", "dpm.args.depth");

	unsigned int const res = vars.getUint32("dpm.args.resolution");
	unsigned int const depth = vars.getUint32("dpm.args.depth");
	DpmShaderGen shaderGen;

	glm::uvec3 r = glm::uvec3(res, res, depth);

	vars.reCreate<Program>("dpm.objects.fillProgram", shaderGen.GetDpmFillProgramShaders(r));
	vars.reCreate<Program>("dpm.objects.shadowMaskProgram", shaderGen.GetDpmTraversalProgramShaders(r));
}

void DeepPrimitiveMap::createShadowMaskVao()
{
	if (!_isValid) return;

	vars.reCreate<VertexArray>("dpm.objects.shadowMaskVao");
}

void DeepPrimitiveMap::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("dpm", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("dpm.objects.shadowMaskFbo");
	fbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	fbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void DeepPrimitiveMap::createLightViewMatrix()
{
	FUNCTION_PROLOGUE("dpm", "lightPosition", "lightUp", "lightView");

	glm::vec3 const pos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));
	glm::vec3 const* up = vars.get<glm::vec3>("lightUp");
	glm::vec3 const* dir = vars.get<glm::vec3>("lightView");

	_lightViewMatrix = glm::lookAt(pos, pos + *dir, *up);
}

void DeepPrimitiveMap::createLightProjMatrix()
{
	FUNCTION_PROLOGUE("dpm", "lightPosition", "dpm.args.near", "dpm.args.far", "dpm.args.fovy", "dpm.args.resolution");

	float nearZ = vars.getFloat("dpm.args.near");
	float farZ = vars.getFloat("dpm.args.far");
	float fovY = vars.getFloat("dpm.args.fovy");

	_lightProjMatrix = glm::perspective(fovY, 1.f, nearZ, farZ);
}

void DeepPrimitiveMap::renderIzb(glm::mat4 const& lightVP)
{
	Buffer* atomicCounter = vars.get<Buffer>("dpm.objects.atomicCounter");
	Program* program = vars.get<Program>("dpm.objects.fillProgram");
	unsigned int const res = vars.getUint32("dpm.args.resolution");

	atomicCounter->clear(GL_R32UI, GL_RED, GL_INT);

	vars.get<Framebuffer>("dpm.objects.FBO")->bind();
	vars.get<VertexArray>("dpm.objects.VAO")->bind();

	glViewport(0, 0, res, res);
	glDisable(GL_DEPTH_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);

	program->use();
	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));

	vars.get<Buffer>("dpm.objects.dpm")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	atomicCounter->bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDrawArrays(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices);

	vars.get<Framebuffer>("dpm.objects.FBO")->unbind();
	vars.get<VertexArray>("dpm.objects.VAO")->unbind();

	glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
}

void DeepPrimitiveMap::createShadowMask(glm::mat4 const& lightVP)
{
	glm::uvec2 windowSize = *vars.get<glm::uvec2>("windowSize");

	glEnable(GL_DEPTH_TEST);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glViewport(0, 0, windowSize.x, windowSize.y);

	vars.get<Framebuffer>("dpm.objects.shadowMaskFbo")->bind();
	vars.get<VertexArray>("dpm.objects.shadowMaskVao")->bind();

	Program* program = vars.get<Program>("dpm.objects.shadowMaskProgram");
	program->use();
	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));
	program->set4fv("lightPos", glm::value_ptr(*vars.get<glm::vec4>("lightPosition")));
	program->set1f("bias", vars.getFloat("dpm.args.bias"));

	vars.get<GBuffer>("gBuffer")->position->bind(0);

	vars.get<Buffer>("dpm.objects.dpm")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("dpm.objects.atomicCounter")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("dpm.objects.triangleBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<Framebuffer>("dpm.objects.shadowMaskFbo")->unbind();
	vars.get<VertexArray>("dpm.objects.shadowMaskVao")->unbind();

	vars.get<Buffer>("dpm.objects.dpm")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("dpm.objects.atomicCounter")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("dpm.objects.triangleBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 2);
}

