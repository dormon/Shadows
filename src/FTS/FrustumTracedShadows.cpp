#include <FrustumTracedShadows.h>
#include <FtsShaderGen.h>

#include <cassert>
#include <algorithm>

#include <geGL/Texture.h>
#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <gegl/Framebuffer.h>
#include <geGL/VertexArray.h>

#include <ifExistStamp.h>
#include <FunctionPrologue.h>
#include <Model.h>
#include <Deferred.h>

using namespace ge;
using namespace gl;

FrustumTracedShadows::FrustumTracedShadows(vars::Vars& vars) : ShadowMethod(vars)
{
	_isValid = IsConservativeRasterizationSupported();

	createShadowMaskVao();
}

FrustumTracedShadows::~FrustumTracedShadows()
{
	vars.erase("fts");
}

void FrustumTracedShadows::create(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	if (!_isValid) return;

	createBuffers();
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

bool FrustumTracedShadows::IsConservativeRasterizationSupported() const
{
    s32 NumberOfExtensions;
    glGetIntegerv(GL_NUM_EXTENSIONS, &NumberOfExtensions);
    for (s32 i = 0; i < NumberOfExtensions; i++) 
    {
        const char* ccc = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));

        if (strcmp(ccc, "GL_NV_conservative_raster") == 0) 
        {
            return true;
        }
    }

    return false;
}

void FrustumTracedShadows::createBuffers()
{
	FUNCTION_PROLOGUE("fts", "args.fts.resolution", "args.fts.depth");

	unsigned int res = vars.getUint32("args.fts.resolution");
	unsigned int depth = vars.getUint32("args.fts.depth");

	vars.reCreate<Buffer>("fts.izb", res * res * depth * sizeof(u32));
	vars.reCreate<Buffer>("fts.atomicCounter", res * res * sizeof(u32));
}

void FrustumTracedShadows::createVao()
{
	FUNCTION_PROLOGUE("fts", "renderModel");

	VertexArray* vao = vars.reCreate<VertexArray>("fts.VAO");

	vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices, 0, 3, GL_FLOAT);
}

void FrustumTracedShadows::createFbo()
{
	FUNCTION_PROLOGUE("fts", "args.fts.resolution");

	unsigned int const res = vars.getUint32("args.fts.resolution");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("fts.FBO");
	fbo->setDefaultWidth(res);
	fbo->setDefaultHeight(res);
}

void FrustumTracedShadows::createShaders()
{
	FUNCTION_PROLOGUE("fts", "args.fts.resolution", "args.fts.depth");

	unsigned int const res = vars.getUint32("args.fts.resolution");
	unsigned int const depth = vars.getUint32("args.fts.depth");
	FtsShaderGen shaderGen;

	glm::uvec3 r = glm::uvec3(res, res, depth);

	vars.reCreate<Program>("fts.fillProgram", shaderGen.GetIzbFillProgramShaders(r));
	vars.reCreate<Program>("fts.shadowMaskProgram", shaderGen.GetIzbTraversalProgramShaders(r));
}

void FrustumTracedShadows::createShadowMaskVao()
{
	if (!_isValid) return;

	vars.reCreate<VertexArray>("fts.shadowMaskVao");
}

void FrustumTracedShadows::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("fts", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("fts.shadowMaskFbo");
	fbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	fbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void FrustumTracedShadows::createLightViewMatrix()
{
	FUNCTION_PROLOGUE("fts", "lightPosition", "lightUp", "lightView");

	glm::vec3 const pos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));
	glm::vec3 const* up = vars.get<glm::vec3>("lightUp");
	glm::vec3 const* dir = vars.get<glm::vec3>("lightView");

	_lightViewMatrix = glm::lookAt(pos, pos + *dir, *up);
}

void FrustumTracedShadows::createLightProjMatrix()
{
	FUNCTION_PROLOGUE("fts", "lightPosition", "args.fts.near", "args.fts.far", "args.fts.fovy", "args.fts.resolution");

	float nearZ = vars.getFloat("args.fts.near");
	float farZ = vars.getFloat("args.fts.far");
	float fovY = vars.getFloat("args.fts.fovy");

	glm::vec4 const pos = *vars.get<glm::vec4>("lightPosition");

	_lightProjMatrix = glm::perspective(fovY, 1.f, nearZ, farZ);
}

void FrustumTracedShadows::renderIzb(glm::mat4 const& lightVP)
{
	Buffer* atomicCounter = vars.get<Buffer>("fts.atomicCounter");
	Program* program = vars.get<Program>("fts.fillProgram");
	unsigned int const res = vars.getUint32("args.fts.resolution");

	atomicCounter->clear(GL_R32UI, GL_RED, GL_INT);

	vars.get<Framebuffer>("fts.FBO")->bind();
	vars.get<VertexArray>("fts.VAO")->bind();

	glViewport(0, 0, res, res);
	glDisable(GL_DEPTH_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);

	program->use();
	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));

	vars.get<Buffer>("fts.izb")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	atomicCounter->bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDrawArrays(GL_TRIANGLES, 0, vars.get<RenderModel>("renderModel")->nofVertices);

	vars.get<Framebuffer>("fts.FBO")->unbind();
	vars.get<VertexArray>("fts.VAO")->unbind();

	glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
	//--
	/*
	glFinish();
	std::vector<u32> vec, izbvec;
	vec.resize(FtsResolution.x * FtsResolution.y);
	izbvec.resize(FtsResolution.x * FtsResolution.y * FtsResolution.z);
	AtomicCounter->getData(vec.data(), FtsResolution.x * FtsResolution.y * sizeof(u32));
	IrregularZBuffer->getData(izbvec.data(), FtsResolution.x * FtsResolution.y * FtsResolution.z * sizeof(u32));
	glFinish();
	u32 minVal = 50000;
	u32 maxVax = 0;
	for (auto const& val : vec)
	{
		if (val != 0)
		{
			minVal = std::min(minVal, val);
			maxVax = std::max(maxVax, val);
		}
	}
	//*/
	//--
}

void FrustumTracedShadows::createShadowMask(glm::mat4 const& lightVP)
{
	glm::uvec2 windowSize = *vars.get<glm::uvec2>("windowSize");

	glEnable(GL_DEPTH_TEST);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glViewport(0, 0, windowSize.x, windowSize.y);

	vars.get<Framebuffer>("fts.shadowMaskFbo")->bind();
	vars.get<VertexArray>("fts.shadowMaskVao")->bind();

	Program* program = vars.get<Program>("fts.shadowMaskProgram");
	program->use();
	program->setMatrix4fv("lightVP", glm::value_ptr(lightVP));
	program->set4fv("lightPos", glm::value_ptr(*vars.get<glm::vec4>("lightPosition")));

	vars.get<GBuffer>("gBuffer")->position->bind(0);

	vars.get<Buffer>("fts.izb")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("fts.atomicCounter")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<RenderModel>("renderModel")->vertices->bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<Framebuffer>("fts.shadowMaskFbo")->unbind();
	vars.get<VertexArray>("fts.shadowMaskVao")->unbind();

	vars.get<Buffer>("fts.izb")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("fts.atomicCounter")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<RenderModel>("renderModel")->vertices->unbindBase(GL_SHADER_STORAGE_BUFFER, 2);
}

