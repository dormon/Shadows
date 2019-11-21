#pragma once

#include <ShadowVolumes.h>

class GSSV : public ShadowVolumes
{
public:
	GSSV(vars::Vars& vars);

	virtual ~GSSV();

	void drawSides(glm::vec4 const&lightPosition, glm::mat4 const&viewMatrix, glm::mat4 const&projectionMatrix) override;
	void drawCaps(glm::vec4 const&lightPosition, glm::mat4 const&viewMatrix, glm::mat4 const&projectionMatrix) override;
    void drawUser(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix) override;

private:
	void createSidesVBO(vars::Vars& vars);
	void createSidesVAO(vars::Vars& vars);
	void createCapsDrawer(vars::Vars& vars);
	void createSidesPrograms(vars::Vars& vars);

	unsigned int getNofAttributes() const;

	size_t	NofEdges = 0;
};
