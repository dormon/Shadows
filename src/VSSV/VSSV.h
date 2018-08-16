#pragma once

#include<geGL/geGL.h>
#include<ShadowVolumes.h>
#include<Model.h>
#include<TimeStamp.h>

#include<Vars/Vars.h>
#include<VSSV/DrawCaps.h>

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
    std::unique_ptr<DrawCaps>caps;
    std::shared_ptr<ge::gl::Program>    drawSidesProgram       = nullptr;
    std::shared_ptr<ge::gl::Buffer>     adjacency              = nullptr;
    std::shared_ptr<ge::gl::VertexArray>sidesVao               = nullptr;
    size_t                              nofEdges               = 0      ;
    size_t                              maxMultiplicity        = 0      ;
    size_t                              nofTriangles           = 0      ;

    void                                createSideDataUsingPoints   (std::shared_ptr<Adjacency const>const&adj);
    void                                createSideDataUsingAllPlanes(std::shared_ptr<Adjacency const>const&adj);
    void                                createSideDataUsingPlanes   (std::shared_ptr<Adjacency const>const&adj);
};
