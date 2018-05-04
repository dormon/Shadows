#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>
#include<Model.h>
#include<FastAdjacency.h>
#include<TimeStamp.h>
#include<RSSVTiles.h>

struct RSSVParams{
    size_t     computeSilhouetteWGS         = 64             ;
    bool       localAtomic                  = true           ;
    bool       cullSides                    = false          ;
    size_t     silhouettesPerWorkgroup      = 1              ;
    bool       usePlanes                    = 0              ;
    glm::uvec2 copyDepthToLastLevelOfHDTWGS = glm::uvec2(8,8);
};

class RSSV: public ShadowMethod{
  public:
    RSSV(
        std::shared_ptr<ge::gl::Texture>const&shadowMask     ,
        glm::uvec2                      const&windowSize     ,
        std::shared_ptr<ge::gl::Texture>const&depthTexture   ,
        std::shared_ptr<Model>          const&model          ,
        size_t                          const&maxMultiplicity,
        RSSVParams                      const&params         ,
        size_t                          const&wavefrontSize  );
    virtual ~RSSV();
    virtual void create(
        glm::vec4 const&lightPosition,
        glm::mat4 const&view         ,
        glm::mat4 const&projection   )override;
  protected:
  public:
    glm::uvec2                      _windowSize                = glm::uvec2(512,512);
    std::shared_ptr<ge::gl::Texture>_shadowMask                = nullptr            ;
    std::shared_ptr<ge::gl::Texture>_depthTexture              = nullptr            ;
    size_t                          _wavefrontSize             = 64                 ;
    std::shared_ptr<ge::gl::Buffer> _triangles                 = nullptr            ;
    std::shared_ptr<ge::gl::Buffer> _edges                     = nullptr            ;
    std::shared_ptr<ge::gl::Buffer> _silhouettes               = nullptr            ;
    size_t                          _nofEdges                  = 0                  ;
    size_t                          _nofTriangles              = 0                  ;
    std::shared_ptr<ge::gl::Program>_computeSilhouettesProgram = nullptr            ;
    std::shared_ptr<ge::gl::Buffer> _dispatchIndirectBuffer    = nullptr            ;
    std::shared_ptr<ge::gl::Program>_generateHDT0Program       = nullptr            ;
    std::shared_ptr<ge::gl::Program>_generateHDTProgram        = nullptr            ;
    size_t                          _nofLevels                 = 1                  ;
    std::vector<std::shared_ptr<ge::gl::Texture>>_HDT;
    std::shared_ptr<ge::gl::Texture>_screenSpaceMultiplicity   = nullptr            ;
    std::shared_ptr<ge::gl::Program>_rasterizeProgram          = nullptr            ;
    RSSVParams                      _params                                         ;
    RSSVTilingSizes                 _tiling                                         ;
    void _generateHDT();
    void _copyDepthToLastLevelOfHDT();
    void _computeAllLevelsOfHDTExceptLast();
    void _computeSilhouettes(glm::vec4 const&lightPosition);
    void _rasterize(glm::vec4 const&lightPosition,glm::mat4 const&view,glm::mat4 const&projection);
    void _allocateHDT();
};
