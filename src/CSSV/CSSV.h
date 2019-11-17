#pragma once

#include <geGL/geGL.h>
#include <glm/glm.hpp>
#include <ShadowVolumes.h>
#include <Model.h>
#include <TimeStamp.h>
#include <Vars/Vars.h>
#include <CSSV/Param.h>
#include <CSSV/DrawCaps.h>
#include <CSSV/DrawSides.h>

class Adjacency;

class cssv::CSSV: public ShadowVolumes{
  public:
    CSSV(vars::Vars&vars);
    virtual ~CSSV();
    virtual void drawSides(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
    virtual void drawCaps(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix)override;
};

