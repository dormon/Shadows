#pragma once

#include<geGL/geGL.h>
#include<ShadowMethod.h>

#include<Vars/Fwd.h>

class CubeShadowMapping: public ShadowMethod{
  public:
    CubeShadowMapping(
        vars::Vars&vars  );
    virtual ~CubeShadowMapping();
    virtual void create(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
    void fillShadowMap(glm::vec4 const&lightPosition);
    void fillShadowMask(glm::vec4 const&lightPosition);
};
