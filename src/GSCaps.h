#pragma once

#include <geGL/VertexArray.h>
#include <geGL/Program.h>
#include <geGL/Buffer.h>

#include <glm/glm.hpp>

#include <Vars/Vars.h>

class Adjacency;

class GSCaps
{
public:
	GSCaps(vars::Vars& vars);
	void drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix);
	void drawCapsVisualized(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix, bool drawFrontCap, bool drawBackCap, bool drawLightFacing, bool drawLightBackFacing, glm::vec3 const& color);

protected:
	void _initCapsBuffers();
	void _initCapsPrograms();

	size_t									_nofCapsTriangles;
	vars::Vars&								vars;
};
