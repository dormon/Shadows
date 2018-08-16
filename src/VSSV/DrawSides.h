#pragma once

#include<glm/glm.hpp>

class DrawSides{
  public:
    virtual void draw(
      glm::vec4 const&lightPosition   ,
      glm::mat4 const&viewMatrix      ,
      glm::mat4 const&projectionMatrix) = 0;
};
