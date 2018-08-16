#pragma once

#include<geGL/geGL.h>
#include<ShadowVolumes.h>
#include<Model.h>
#include<TimeStamp.h>

#include<Vars/Vars.h>

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
    std::shared_ptr<ge::gl::Program>    drawSidesProgram       = nullptr;
    std::shared_ptr<ge::gl::Buffer>     adjacency              = nullptr;
    std::shared_ptr<ge::gl::VertexArray>sidesVao               = nullptr;
    size_t                              nofEdges               = 0      ;
    size_t                              maxMultiplicity        = 0      ;
    size_t                              nofTriangles           = 0      ;

    std::shared_ptr<ge::gl::Program>    drawCapsProgram        = nullptr;
    std::shared_ptr<ge::gl::VertexArray>capsVao                = nullptr;
    std::shared_ptr<ge::gl::Buffer>     caps                   = nullptr;

    void                                createSideDataUsingPoints   (std::shared_ptr<Adjacency const>const&adj);
    void                                createSideDataUsingAllPlanes(std::shared_ptr<Adjacency const>const&adj);
    void                                createSideDataUsingPlanes   (std::shared_ptr<Adjacency const>const&adj);
    void                                createCapDataUsingPoints    (std::shared_ptr<Adjacency const>const&adj);
};
