#pragma once

#include<Vars/Vars.h>
#include<glm/glm.hpp>
#include<geGL/geGL.h>
#include<CSSV/Fwd.h>


class Adjacency;
class cssv::ExtractSilhouettes{
  public:
    ExtractSilhouettes(vars::Vars&vars);
    virtual ~ExtractSilhouettes(){}
    virtual void compute(glm::vec4 const&lightPosition);
    vars::Vars                      &vars        ;
};

