#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>
#include<Model.h>
#include<FastAdjacency.h>
#include<TimeStamp.h>
#include<RSSVTiles.h>

#include<RSSVParam.h>
#include<Vars.h>

class RSSV: public ShadowMethod{
  public:
    RSSV(vars::Vars&vars);
    virtual ~RSSV();
    virtual void create(
        glm::vec4 const&lightPosition,
        glm::mat4 const&view         ,
        glm::mat4 const&projection   )override;
  protected:
  public:
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
    RSSVTilingSizes                 _tiling                                         ;
    vars::Vars&vars;
    void _generateHDT();
    void _copyDepthToLastLevelOfHDT();
    void _computeAllLevelsOfHDTExceptLast();
    void _computeSilhouettes(glm::vec4 const&lightPosition);
    void _rasterize(glm::vec4 const&lightPosition,glm::mat4 const&view,glm::mat4 const&projection);
    void _allocateHDT();
};
