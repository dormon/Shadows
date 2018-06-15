#pragma once

#include<geGL/geGL.h>
#include<glm/glm.hpp>
#include<ShadowVolumes.h>
#include<Model.h>
#include<TimeStamp.h>
#include<Vars.h>
#include <CSSVParam.h>

class Adjacency;
class CSSV: public ShadowVolumes{
  public:
    CSSV(
        vars::Vars                      const&vars           ,
        CSSVParams                      const&params         );
    virtual ~CSSV();
    virtual void drawSides(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
    virtual void drawCaps(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
    std::shared_ptr<ge::gl::Program>    _computeSidesProgram = nullptr;
    std::shared_ptr<ge::gl::Program>    _drawSidesProgram    = nullptr;
    std::shared_ptr<ge::gl::Buffer>     _edges               = nullptr;
    std::shared_ptr<ge::gl::Buffer>     _sillhouettes        = nullptr;
    std::shared_ptr<ge::gl::Buffer>     _dibo                = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_sidesVao            = nullptr;
    size_t                              _nofEdges            = 0      ;
    size_t                              _nofTriangles        = 0      ;
    CSSVParams                          _params                       ;

    std::shared_ptr<ge::gl::Program>    _drawCapsProgram  = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_capsVao          = nullptr;
    std::shared_ptr<ge::gl::Buffer>     _caps             = nullptr;
    vars::Vars const&vars;

    void _computeSides(glm::vec4 const&lightPosition);

    void _createSidesData                           (std::shared_ptr<Adjacency const>const&adj);
    void _createSidesDataUsingPlanes                (std::shared_ptr<Adjacency const>const&adj);
    void _createSidesDataUsingPlanesWithInterleaving(std::shared_ptr<Adjacency const>const&adj);
    void _createCapsData                            (std::shared_ptr<Adjacency const>const&adj);
};
