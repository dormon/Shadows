#pragma once

#include<geGL/geGL.h>
#include<ShadowVolumes.h>
#include<Model.h>
#include<TimeStamp.h>

#include<Vars.h>

class Adjacency;
class VSSV: public ShadowVolumes{
  public:
    VSSV(vars::Vars&vars);
    virtual ~VSSV();
    virtual void drawSides(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
    virtual void drawCaps(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
    std::shared_ptr<ge::gl::Program>    _drawSidesProgram       = nullptr;
    std::shared_ptr<ge::gl::Buffer>     _adjacency              = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_sidesVao               = nullptr;
    size_t                              _nofEdges               = 0      ;
    size_t                              _maxMultiplicity        = 0      ;
    size_t                              _nofTriangles           = 0      ;

    std::shared_ptr<ge::gl::Program>    _drawCapsProgram        = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_capsVao                = nullptr;
    std::shared_ptr<ge::gl::Buffer>     _caps                   = nullptr;

    void                                _createSideDataUsingPoints   (std::shared_ptr<Adjacency>const&adj);
    void                                _createSideDataUsingAllPlanes(std::shared_ptr<Adjacency>const&adj);
    void                                _createSideDataUsingPlanes   (std::shared_ptr<Adjacency>const&adj);
    void                                _createCapDataUsingPoints    (std::shared_ptr<Adjacency>const&adj);
};
