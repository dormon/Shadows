#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>

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
    std::shared_ptr<ge::gl::Framebuffer>_fbo         = nullptr;
    std::shared_ptr<ge::gl::Framebuffer>_maskFbo     = nullptr;
    std::shared_ptr<ge::gl::Program>    _blitProgram = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_emptyVao    = nullptr;
    void _blit();
};
