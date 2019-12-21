#include <DPSV.h>
#include <DpsvShaders.h>

#include <geGL/Texture.h>
#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/Framebuffer.h>
#include <geGL/VertexArray.h>

#include <ifExistStamp.h>
#include <FunctionPrologue.h>
#include <Model.h>
#include <Deferred.h>

using namespace ge::gl;

DPSV::DPSV(vars::Vars& vars) : ShadowMethod(vars)
{
	createTraversalShaders();
	createAuxBuffer();
	createShadowMaskVao();
}

DPSV::~DPSV()
{
	vars.erase("dpsv.objects");
}

void DPSV::create(glm::vec4 const& lightPosition, glm::mat4 const&, glm::mat4 const&)
{
	createBuildShader();
	createNodeBuffer();
	createShadowMaskFbo();

	ifExistStamp("");
	buildTopTree(lightPosition);
	ifExistStamp("build");

	createShadowMask(lightPosition);
	ifExistStamp("traverse");
}

void DPSV::createAuxBuffer()
{
	vars.reCreate<Buffer>("dpsv.objects.auxBuffer", 4 * sizeof(uint32_t));
}

void DPSV::createNodeBuffer()
{
	FUNCTION_PROLOGUE("dpsv.objects", "renderModel");

	size_t const nodeSize = 4 * sizeof(float) + 4 * sizeof(uint32_t); //Node struct on the GPU
	size_t const nofVertices = vars.get<RenderModel>("renderModel")->nofVertices;

	vars.reCreate<Buffer>("dpsv.objects.nodeBuffer", 4 * (nofVertices + 1) * nodeSize);

	NofTriangles = uint32_t(nofVertices) / 3;
}

void DPSV::createBuildShader()
{
	FUNCTION_PROLOGUE("dpsv.objects", "dpsv.args.wgSize");

	uint32_t wgSize = vars.getUint32("dpsv.args.wgSize");

	vars.reCreate<Program>("dpsv.objects.buildCS", getDpsvBuildCS(wgSize));
}

void DPSV::createTraversalShaders()
{
	vars.reCreate<Program>("dpsv.objects.traverseStack", getDpsvStackProgramShaders());
	vars.reCreate<Program>("dpsv.objects.traverseStackless", getDpsvStacklessProgramShaders());
	vars.reCreate<Program>("dpsv.objects.traverseHybrid", getDpsvHybridProgramShaders());
}

void DPSV::createShadowMaskFbo()
{
	FUNCTION_PROLOGUE("dpsv.objects", "shadowMask");

	Framebuffer* fbo = vars.reCreate<Framebuffer>("dpsv.objects.FBO");
	fbo->attachTexture(GL_COLOR_ATTACHMENT0, vars.get<Texture>("shadowMask"));
	fbo->drawBuffers(1, GL_COLOR_ATTACHMENT0);
}

void DPSV::createShadowMaskVao()
{
	vars.reCreate<VertexArray>("dpsv.objects.VAO");
}

void DPSV::clearAuxBuffer()
{
	Buffer* buffer = vars.get<Buffer>("dpsv.objects.auxBuffer");
	
	glm::uvec4 const data = glm::uvec4(0, 4, 0, 0);
	buffer->setData(glm::value_ptr(data));
}

void DPSV::setWindowViewport()
{
	glm::uvec2 const windowSize = *vars.get<glm::uvec2>("windowSize");
	glViewport(0, 0, windowSize.x, windowSize.y);
}

void DPSV::buildTopTree(glm::vec4 const& lightPosition)
{
	clearAuxBuffer();
	
	Program* program = vars.get<Program>("dpsv.objects.buildCS");
	program->use();
	program->set4fv("lightPosition", glm::value_ptr(lightPosition));
	program->set1ui("nofTriangles", NofTriangles);
	program->set1f("bias", vars.getFloat("dpsv.args.bias"));
	
	vars.get<RenderModel>("renderModel")->vertices->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("dpsv.objects.nodeBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("dpsv.objects.auxBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	glDispatchCompute(vars.getUint32("dpsv.args.numWg"), 1, 1);
	
	vars.get<RenderModel>("renderModel")->vertices->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("dpsv.objects.nodeBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);
	vars.get<Buffer>("dpsv.objects.auxBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 2);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void DPSV::createShadowMask(glm::vec4 const& lightPosition)
{
	vars.get<VertexArray>("dpsv.objects.VAO")->bind();
	vars.get<Framebuffer>("dpsv.objects.FBO")->bind();
	setWindowViewport();

	Program* program = selectTraversalProgram();
	program->use();
	program->set4fv("lightPosition", glm::value_ptr(lightPosition));
	
	vars.get<GBuffer>("gBuffer")->position->bind(0);

	vars.get<Buffer>("dpsv.objects.nodeBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("dpsv.objects.auxBuffer")->bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	vars.get<Buffer>("dpsv.objects.nodeBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 0);
	vars.get<Buffer>("dpsv.objects.auxBuffer")->unbindBase(GL_SHADER_STORAGE_BUFFER, 1);

	vars.get<VertexArray>("dpsv.objects.VAO")->unbind();
	vars.get<Framebuffer>("dpsv.objects.FBO")->unbind();
}

ge::gl::Program* DPSV::selectTraversalProgram() const
{
	uint32_t const alg = vars.getUint32("dpsv.args.algVersion");

	switch(alg)
	{
	case 1:
		return vars.get<Program>("dpsv.objects.traverseStackless");
	case 2:
		return vars.get<Program>("dpsv.objects.traverseHybrid");
	default:
	case 0:
		return vars.get<Program>("dpsv.objects.traverseStack");
	}
}


