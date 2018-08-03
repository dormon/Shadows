#pragma once

#include<memory>
#include<glm/glm.hpp>
#include<geGL/geGL.h>
#include<Vars/Vars.h>
#include<FastAdjacency.h>
#include<CSSV/Fwd.h>

class cssv::DrawCaps{
  public:
    std::shared_ptr<ge::gl::Buffer     >caps   ;
    std::shared_ptr<ge::gl::VertexArray>vao    ;
    std::shared_ptr<ge::gl::Program    >program;
    size_t nofTriangles = 0;
    DrawCaps(std::shared_ptr<Adjacency const>const&adj);
    void draw(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix);
};
