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
    bool       _useUniformTileSizeInClipSpace;
    bool       _useUniformTileDivisibility;

    std::shared_ptr<ge::gl::Texture>_finalStencilMask;
    
    std::shared_ptr<ge::gl::VertexArray>_emptyVao;
    std::vector<std::shared_ptr<ge::gl::Texture>>_HST;

    std::shared_ptr<ge::gl::Program>RasterizeTextureProgram;
    std::shared_ptr<ge::gl::Program>ClearStencilProgram;

    ge::gl::Texture*_shadowMask;
    std::shared_ptr<ge::gl::Program>_blitProgram;

    std::shared_ptr<ge::gl::Program>_drawHSTProgram;
    void drawHST(size_t l);
    std::shared_ptr<ge::gl::Program>_drawFinalStencilMask;
    void drawFinalStencilMask();

    void RasterizeTexture();
    void MergeTexture();
    void blit();
};
