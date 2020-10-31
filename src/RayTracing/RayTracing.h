#pragma once

#include<ShadowMethod.h>

#include<Vars/Fwd.h>

class RayTracing: public ShadowMethod{
  public:
    RayTracing(
        vars::Vars&vars  );
    virtual ~RayTracing();
    virtual void create(
        glm::vec4 const&lightPosition   ,
        glm::mat4 const&viewMatrix      ,
        glm::mat4 const&projectionMatrix)override;
  protected:
};
