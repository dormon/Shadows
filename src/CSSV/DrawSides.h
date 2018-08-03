#pragma once

#include<memory>
#include<glm/glm.hpp>
#include<geGL/geGL.h>
#include<CSSV/Fwd.h>

class cssv::DrawSides{
  public:
    std::shared_ptr<ge::gl::Program    >program;
    std::shared_ptr<ge::gl::VertexArray>vao    ;
    std::shared_ptr<ge::gl::Buffer     >dibo   ;
    DrawSides(std::shared_ptr<ge::gl::Buffer>const&silhouettes,std::shared_ptr<ge::gl::Buffer>const&dibo);
    void draw(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix);
};
