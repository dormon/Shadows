#include "GSCaps.h"
#include "GSCapsShaders.h"
#include "geGL/StaticCalls.h"

#include <glm/gtc/type_ptr.hpp>

GSCaps::GSCaps(std::shared_ptr<Adjacency const> adjacency)
{
	_initCapsBuffers(adjacency);
	_initCapsProgram();
}

void GSCaps::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	const glm::mat4 mvp = projectionMatrix * viewMatrix;

	_capsVAO->bind();
	_capsProgram->use();
	_capsProgram->setMatrix4fv("mvp", glm::value_ptr(mvp), 1, GL_FALSE);
	_capsProgram->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);
	ge::gl::glDrawArrays(GL_TRIANGLES, 0, GLsizei(_nofCapsTriangles * 3));

	_capsVAO->unbind();
}

void GSCaps::drawCapsVisualized(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix, bool drawFrontCap, bool drawBackCap, bool drawLightFacing, bool drawLightBackFacing, glm::vec3 const& color)
{
	const glm::mat4 mvp = projectionMatrix * viewMatrix;

	_capsVAO->bind();

	_capsVisualizationProgram->use();
	_capsVisualizationProgram->setMatrix4fv("mvp", glm::value_ptr(mvp), 1, GL_FALSE);
	_capsVisualizationProgram->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);
	_capsVisualizationProgram->set1i("drawFrontCap", drawFrontCap ? 1 : 0);
	_capsVisualizationProgram->set1i("drawBackCap", drawBackCap ? 1 : 0);
	_capsVisualizationProgram->set1i("drawLightFace", drawLightFacing ? 1 : 0);
	_capsVisualizationProgram->set1i("drawLightBackFace", drawLightBackFacing ? 1 : 0);
	_capsVisualizationProgram->set3fv("color", glm::value_ptr(color));

	ge::gl::glDrawArrays(GL_TRIANGLES, 0, GLsizei(_nofCapsTriangles * 3));

	_capsVAO->unbind();
}

void GSCaps::_initCapsBuffers(std::shared_ptr<Adjacency const> ad)
{
	_capsVAO = std::make_shared<ge::gl::VertexArray>();

	_nofCapsTriangles = ad->getNofTriangles();

	_capsVBO = std::make_shared<ge::gl::Buffer>(sizeof(float) * 4 * 3 * _nofCapsTriangles, nullptr, GL_STATIC_DRAW);

	float*Ptr = (float*)_capsVBO->map();
	for (unsigned t = 0; t<_nofCapsTriangles; ++t)
	{
		for (unsigned p = 0; p<3; ++p) {
			for (unsigned i = 0; i<3; ++i)
				Ptr[(t * 3 + p) * 4 + i] = ad->getVertices()[(t * 3 + p) * 3 + i];
			Ptr[(t * 3 + p) * 4 + 3] = 1;
		}
	}

	_capsVBO->unmap();

	_capsVAO->addAttrib(_capsVBO, 0, 4, GL_FLOAT);
}

void GSCaps::_initCapsProgram()
{
	std::shared_ptr<ge::gl::Shader> vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, vsSource);
	std::shared_ptr<ge::gl::Shader> gs = std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER, gsSource);
	std::shared_ptr<ge::gl::Shader> fs = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fsSource);

	_capsProgram = std::make_shared<ge::gl::Program>(vs, gs, fs);

	_capsVisualizationProgram = std::make_shared<ge::gl::Program>(vs, 
		std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER, gsSourceVisualize),
		std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fsSourceVisualize));
}
