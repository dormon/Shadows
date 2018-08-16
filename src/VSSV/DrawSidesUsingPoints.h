#pragma once

#include <VSSV/DrawSides.h>
#include <geGL/geGL.h>
#include <Vars/Vars.h>

class Adjacency;
class DrawSidesUsingPoints: public DrawSides{
  public:
    DrawSidesUsingPoints(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
    virtual void draw(
      glm::vec4 const&lightPosition   ,
      glm::mat4 const&viewMatrix      ,
      glm::mat4 const&projectionMatrix) override;
  protected:
    vars::Vars&                         vars               ;
    std::shared_ptr<ge::gl::Program>    program            ;
    std::shared_ptr<ge::gl::Buffer>     sides              ;
    std::shared_ptr<ge::gl::VertexArray>vao                ;
    size_t                              nofEdges        = 0;
    size_t                              maxMultiplicity = 0;
};
