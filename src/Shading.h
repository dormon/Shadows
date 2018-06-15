#pragma once

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtc/matrix_access.hpp>

#include<geGL/geGL.h>
#include<Vars.h>

class Shading: ge::gl::Context{
  public:
    Shading(vars::Vars const&vars);
    ~Shading();
    void draw(glm::vec4 const&lightPosition,glm::vec3 const&cameraPosition,bool useShadows);
  protected:
    vars::Vars const&vars;
    std::shared_ptr<ge::gl::Program>_program    = nullptr;
    std::shared_ptr<ge::gl::VertexArray>_vao    = nullptr;
};
