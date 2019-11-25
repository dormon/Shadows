#pragma once

#include <SidesDrawerBase.h>

#include <geGL/Buffer.h>
#include <geGL/Program.h>
#include <geGL/VertexArray.h>

class Adjacency;

class GpuSidesDrawer : public SidesDrawerBase
{
public:
	GpuSidesDrawer(Octree* octree, Adjacency* ad, u32 maxMultiplicity);
	~GpuSidesDrawer();

	void drawSides(const glm::mat4& mvp, const glm::vec4& light) override;

private:
	void CreateShaders();
	void CreateBuffers();
	void LoadEdgeBuffer();

private:
	std::unique_ptr<ge::gl::Program> getBufferIdsSizesCs;
	std::unique_ptr<ge::gl::Program> generateSidesCs;
	std::unique_ptr<ge::gl::Program> drawSidesProgram;

	std::unique_ptr<ge::gl::Buffer> edgesBuffer;
	std::unique_ptr<ge::gl::Buffer> octreeBuffer;

	Adjacency* Ad;
	u32 MaxMultiplicity;
};