#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>
#include<FastAdjacency.h>

class ShadowVolumes: public ShadowMethod{
  public:
    ShadowVolumes(vars::Vars&vars);
    virtual ~ShadowVolumes();
    virtual void create(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override final;
    virtual void drawSides(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix) = 0;
    virtual void drawCaps(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix) = 0;
  protected:
    void convertStencilBufferToShadowMask();
};

std::shared_ptr<Adjacency const> createAdjacency(vars::Vars&vars);
