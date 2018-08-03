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
    std::shared_ptr<ge::gl::Framebuffer>fbo         = nullptr;
    std::shared_ptr<ge::gl::Framebuffer>maskFbo     = nullptr;
    std::shared_ptr<ge::gl::Program    >blitProgram = nullptr;
    std::shared_ptr<ge::gl::VertexArray>emptyVao    = nullptr;
    void blit();
};

std::shared_ptr<Adjacency const> createAdjacency(vars::Vars&vars);
