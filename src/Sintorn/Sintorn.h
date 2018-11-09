#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>
#include<Model.h>

#include<Vars/Vars.h>

class Sintorn: public ShadowMethod{
  public:
    Sintorn(vars::Vars&vars);
    virtual ~Sintorn();
    virtual void create(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
  public:
    std::shared_ptr<ge::gl::VertexArray>_emptyVao;

    ge::gl::Texture*_shadowMask;
    std::shared_ptr<ge::gl::Program>_blitProgram;

    std::shared_ptr<ge::gl::Program>_drawHSTProgram;
    void drawHST(size_t l);
    std::shared_ptr<ge::gl::Program>_drawFinalStencilMask;
    void drawFinalStencilMask();

    void blit();
};
