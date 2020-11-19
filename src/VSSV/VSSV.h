#pragma once

#include<geGL/geGL.h>
#include<ShadowVolumes.h>
#include<Model.h>
#include<TimeStamp.h>

#include<Vars/Fwd.h>
#include<VSSV/DrawCaps.h>
#include<VSSV/DrawSides.h>

class Adjacency;
class VSSV: public ShadowVolumes{
  public:
    VSSV(vars::Vars&vars);
    virtual ~VSSV();
    virtual void drawSides(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
    virtual void drawCaps(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatirx      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
    std::unique_ptr<DrawCaps>caps;
    std::unique_ptr<DrawSides>sides;
};
