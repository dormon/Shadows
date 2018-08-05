#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>

#include<Vars/Vars.h>

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
    void fillShadowMap(glm::vec4 const&lightPosition);
    void fillShadowMask(glm::vec4 const&lightPosition);
    std::shared_ptr<ge::gl::Texture>    shadowMap       ;
    std::shared_ptr<ge::gl::Framebuffer>shadowMapFBO    ;
    std::shared_ptr<ge::gl::VertexArray>shadowMapVAO    ;
    std::shared_ptr<ge::gl::Program>    createShadowMap ;
    std::shared_ptr<ge::gl::VertexArray>maskVao         ;
    std::shared_ptr<ge::gl::Framebuffer>maskFbo         ;
    std::shared_ptr<ge::gl::Program>    createShadowMask;
};
