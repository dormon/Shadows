#pragma once

#include <ShadowVolumes.h>
#include <Vars/Fwd.h>
#include <CSSV/Fwd.h>
#include <CSSV/Param.h>

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

