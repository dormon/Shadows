#pragma once

#include<glm/glm.hpp>
#include<geGL/geGL.h>
#include<Vars/Vars.h>

class Adjacency;
class DrawCaps{
  public:
    DrawCaps(std::shared_ptr<Adjacency const>const&adj);
    void draw(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix);
  protected:
    size_t                              nofTriangles = 0;
    std::shared_ptr<ge::gl::Program>    program         ;
    std::shared_ptr<ge::gl::VertexArray>vao             ;
    std::shared_ptr<ge::gl::Buffer>     caps            ;
};
