#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>

#include<Vars.h>

class CubeShadowMapping: public ShadowMethod{
  public:
    CubeShadowMapping(
        vars::Vars&vars  );
    virtual ~CubeShadowMapping();
    virtual void create(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
    uint32_t                            _nofVertices      = 0      ;
    std::shared_ptr<ge::gl::Texture>    _shadowMap        = nullptr;
    std::shared_ptr<ge::gl::Framebuffer>_fbo              = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_vao              = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_maskVao          = nullptr;
    std::shared_ptr<ge::gl::Framebuffer>_maskFbo          = nullptr;
    std::shared_ptr<ge::gl::Program>    _createShadowMap  = nullptr;
    std::shared_ptr<ge::gl::Program>    _createShadowMask = nullptr;
    vars::Vars&vars;
};
