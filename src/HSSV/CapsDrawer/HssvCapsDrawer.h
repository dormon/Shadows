#pragma once

#include <geGL/VertexArray.h>
#include <geGL/Program.h>
#include <geGL/Buffer.h>

#include <glm/glm.hpp>

class Adjacency;

class HssvCapsDrawer
{
public:
	HssvCapsDrawer(Adjacency* ad);
	~HssvCapsDrawer();

	void drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix);

protected:
	void InitCapsBuffers();
	void InitCapsPrograms();

	std::unique_ptr<ge::gl::VertexArray> VAO;
	std::shared_ptr<ge::gl::Buffer> VBO;
	std::unique_ptr<ge::gl::Program> SidesProgram;

	size_t NofCapsTriangles;
	Adjacency* Ad;
};