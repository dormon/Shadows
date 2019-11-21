#pragma once

#include <ShadowMethod.h>

namespace ge
{
	namespace gl
	{
		class Program;
	}
}

//Deep Partitioned Shadow Volumes
class DPSV : public ShadowMethod
{
public:
	DPSV(vars::Vars& vars);

	virtual ~DPSV();

	virtual void create(glm::vec4 const& lightPosition,
		glm::mat4 const& viewMatrix,
		glm::mat4 const& projectionMatrix) override;

protected:
	void createAuxBuffer();
	void createNodeBuffer();
	void createBuildShader();
	void createTraversalShaders();
	void createShadowMaskFbo();
	void createShadowMaskVao();

	void clearAuxBuffer();
	void setWindowViewport();

	void buildTopTree(glm::vec4 const& lightPosition);
	void createShadowMask(glm::vec4 const& lightPosition);
	ge::gl::Program* selectTraversalProgram() const;

	uint32_t NofTriangles = 0;
};