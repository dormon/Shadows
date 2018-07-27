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
    size_t     _nofLevels = 0;///number of level
    bool       _useUniformTileSizeInClipSpace;
    bool       _useUniformTileDivisibility;
    size_t     _nofTriangles = 0;
    size_t     _wavefrontSize = 64;

    std::shared_ptr<ge::gl::Texture>_finalStencilMask;
    std::shared_ptr<ge::gl::Buffer>_shadowFrusta = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_emptyVao;
    std::shared_ptr<ge::gl::Buffer>_triangles = nullptr;
    std::vector<std::shared_ptr<ge::gl::Texture>>_HDT;
    std::vector<std::shared_ptr<ge::gl::Texture>>_HST;

    std::vector<glm::uvec2>_tileCount;
    std::vector<glm::uvec2>_usedTiles;
    std::vector<glm::uvec2>_tileSizeInPixels;///size of tiles in pixels in each level
    std::vector<glm::uvec2>_tileDivisibility;
    std::vector<glm::vec2>_tileSizeInClipSpace;

    std::shared_ptr<ge::gl::Program>WriteDepthTextureProgram;
    std::shared_ptr<ge::gl::Program>HierarchicalDepthTextureProgram;
    std::shared_ptr<ge::gl::Program>ShadowFrustumProgram;
    std::shared_ptr<ge::gl::Program>RasterizeTextureProgram;
    std::shared_ptr<ge::gl::Program>MergeTextureProgram;
    std::shared_ptr<ge::gl::Program>WriteStencilTextureProgram;
    std::shared_ptr<ge::gl::Program>ClearStencilProgram;
    std::shared_ptr<ge::gl::Program>SFProgram;

    ge::gl::Texture*_shadowMask;
    std::shared_ptr<ge::gl::Program>_blitProgram;

    std::shared_ptr<ge::gl::Program>_drawHSTProgram;
    void drawHST(size_t l);
    std::shared_ptr<ge::gl::Program>_drawFinalStencilMask;
    void drawFinalStencilMask();

    void GenerateHierarchyTexture(glm::vec4 const&lightPosition);
    void ComputeShadowFrusta(glm::vec4 const&lightPosition,glm::mat4 mvp);
    void RasterizeTexture();
    void MergeTexture();
    void blit();
};
