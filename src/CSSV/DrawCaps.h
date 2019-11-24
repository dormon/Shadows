#pragma once

#include<memory>
#include<glm/glm.hpp>
#include<geGL/geGL.h>
#include<Vars/Vars.h>
#include<FastAdjacency.h>
#include<CSSV/Fwd.h>
#include<Vars/Vars.h>

class cssv::DrawCaps{
  public:
    vars::Vars&vars;
    DrawCaps(Adjacency const*adj,vars::Vars&vars);
    void draw(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix);
};
