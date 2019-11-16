#pragma once

#include <geGL/VertexArray.h>
#include <geGL/Program.h>
#include <geGL/Buffer.h>
#include "FastAdjacency.h"

#include <glm/glm.hpp>

class GSCaps
{
public:
	GSCaps(std::shared_ptr<Adjacency const> adjacency);
	void drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix);
	void drawCapsVisualized(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix, bool drawFrontCap, bool drawBackCap, bool drawLightFacing, bool drawLightBackFacing, glm::vec3 const& color);
protected:
	void _initCapsBuffers(std::shared_ptr<Adjacency const> ad);
	void _initCapsProgram();

	std::shared_ptr<ge::gl::Buffer>			_capsVBO;
	std::shared_ptr<ge::gl::VertexArray>	_capsVAO;
	std::shared_ptr<ge::gl::Program>		_capsProgram;
	std::shared_ptr<ge::gl::Program>		_capsVisualizationProgram;

	size_t									_nofCapsTriangles;
};
