#include <GSCaps.h>
#include <GSCapsShaders.h>
#include <geGL/StaticCalls.h>
#include <FastAdjacency.h>
#include <FunctionPrologue.h>

#include <glm/gtc/type_ptr.hpp>

using namespace ge::gl;

GSCaps::GSCaps(vars::Vars& vars) :vars(vars)
{
	_initCapsPrograms();
}

void GSCaps::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	_initCapsBuffers();

	const glm::mat4 mvp = projectionMatrix * viewMatrix;

	vars.get<VertexArray>("gsCaps.VAO")->bind();
	
	Program* program = vars.get<Program>("gsCaps.program");

	program->use();
	program->setMatrix4fv("mvp", glm::value_ptr(mvp), 1, GL_FALSE);
	program->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);

	ge::gl::glDrawArrays(GL_TRIANGLES, 0, GLsizei(_nofCapsTriangles * 3));

	vars.get<VertexArray>("gsCaps.VAO")->unbind();
}

void GSCaps::drawCapsVisualized(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix, bool drawFrontCap, bool drawBackCap, bool drawLightFacing, bool drawLightBackFacing, glm::vec3 const& color)
{
	const glm::mat4 mvp = projectionMatrix * viewMatrix;

	vars.get<VertexArray>("gsCaps.VAO")->bind();

	Program* program = vars.get<Program>("gsCaps.visualizationProgram");
	
	program->use();
	program->setMatrix4fv("mvp", glm::value_ptr(mvp), 1, GL_FALSE);
	program->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);
	program->set1i("drawFrontCap", drawFrontCap ? 1 : 0);
	program->set1i("drawBackCap", drawBackCap ? 1 : 0);
	program->set1i("drawLightFace", drawLightFacing ? 1 : 0);
	program->set1i("drawLightBackFace", drawLightBackFacing ? 1 : 0);
	program->set3fv("color", glm::value_ptr(color));

	ge::gl::glDrawArrays(GL_TRIANGLES, 0, GLsizei(_nofCapsTriangles * 3));

	vars.get<VertexArray>("gsCaps.VAO")->unbind();
}

void GSCaps::_initCapsBuffers()
{
	FUNCTION_PROLOGUE("gsCaps", "adjacency");

	VertexArray* vao = vars.reCreate<VertexArray>("gsCaps.VAO");

	Adjacency const* ad = vars.get<Adjacency>("adjacency");
	_nofCapsTriangles = ad->getNofTriangles();

	Buffer* vbo = vars.reCreate<Buffer>("gsCaps.VBO", sizeof(float) * 4 * 3 * _nofCapsTriangles, nullptr, GL_STATIC_DRAW);

	float*Ptr = (float*)vbo->map();
	for (unsigned t = 0; t<_nofCapsTriangles; ++t)
	{
		for (unsigned p = 0; p<3; ++p) {
			for (unsigned i = 0; i<3; ++i)
				Ptr[(t * 3 + p) * 4 + i] = ad->getVertices()[(t * 3 + p) * 3 + i];
			Ptr[(t * 3 + p) * 4 + 3] = 1;
		}
	}

	vbo->unmap();

	vao->addAttrib(vbo, 0, 4, GL_FLOAT);
}

void GSCaps::_initCapsPrograms()
{
	std::shared_ptr<ge::gl::Shader> vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, vsSource);
	std::shared_ptr<ge::gl::Shader> gs = std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER, gsSource);
	std::shared_ptr<ge::gl::Shader> fs = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fsSource);

	vars.reCreate<Program>("gsCaps.program", vs, gs, fs);

	vars.reCreate<Program>("gsCaps.visualizationProgram", vs,
		std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER, gsSourceVisualize),
		std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fsSourceVisualize));
}
