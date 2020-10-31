#pragma once

#include<Vars/Fwd.h>
#include<geGL/geGL.h>
#include<glm/glm.hpp>

class DrawSides{
  public:
    DrawSides(vars::Vars&vars);
    virtual void draw(
      glm::vec4 const&lightPosition   ,
      glm::mat4 const&viewMatrix      ,
      glm::mat4 const&projectionMatrix);
  protected:
    vars::Vars&                         vars               ;
    std::shared_ptr<ge::gl::Program>    program            ;
    std::shared_ptr<ge::gl::VertexArray>vao                ;
    std::shared_ptr<ge::gl::Buffer>     sides              ;
    size_t                              nofEdges        = 0;
    size_t                              maxMultiplicity = 0;
};
