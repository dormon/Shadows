#pragma once

#include <ShadowVolumes.h>
#include <AABB.h>

class Octree;

class HSSV : public ShadowVolumes
{
public:
	HSSV(vars::Vars& vars);
	~HSSV();

	void drawSides(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix) override;
	void drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix) override;
	void drawUser(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix) override;

protected:
	void getOctree();
	void createCapsDrawer();
	void resetMultiplicity();
	void createSidesDrawer();

	void buildOctree();
	bool loadOctreeFromFile();
	void storeOctree();

	AABB createOctreeVolume() const;
	AABB getSceneAabb() const;
	void fixVolume(AABB& volume) const;

	void buildTest();

};