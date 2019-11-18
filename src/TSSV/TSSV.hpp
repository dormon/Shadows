#pragma once

#include <ShadowVolumes.h>

class TSSV : public ShadowVolumes
{
  public:
	TSSV(vars::Vars& vars);
    
	virtual ~TSSV();
	
	void drawSides( glm::vec4 const&lightPosition, glm::mat4 const&viewMatrix, glm::mat4 const&projectionMatrix) override;
	void drawCaps(glm::vec4 const&lightPosition, glm::mat4 const&viewMatrix, glm::mat4 const&projectionMatrix) override;

private:
	void createVertexBuffer();
	void createElementBuffer();
	void createVertexArray();
	void createProgram();
	void createCapsDrawer();
	void setCounts();

	size_t getNofPatchVertices() const;

	size_t	_patchVertices = 0;
    size_t	_nofEdges = 0;

};
