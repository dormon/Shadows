#pragma once

#include <ShadowMethod.h>

//Metric tree shadow volumes
//Deves 2019
class MTSV : public ShadowMethod
{
public:
	MTSV(vars::Vars& vars);
	virtual ~MTSV();

	virtual void create(glm::vec4 const& lightPosition,
		glm::mat4 const& viewMatrix,
		glm::mat4 const& projectionMatrix) override;

protected:
	void getNofTriangles();
	void createBuildProgram();
	void createTraversalProgram();

	void createNodeBuffer();
	void createTriangleBuffer();
	void createSupportBuffer();
	void clearSupportBuffer();

	void createShadowMaskVao();
	void createShadowMaskFbo();
	void setWindowViewport();

	void buildMetricTree(glm::vec3 const& lightPos);
	void fillShadowMask(glm::vec3 const& lightPo);

protected:
	uint32_t NofTriangles = 0;
};