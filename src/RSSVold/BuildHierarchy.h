#pragma once

#include<Vars/Vars.h>

namespace rssv{

class BuildHierarchy{
  public:
    BuildHierarchy(vars::Vars&vars):vars(vars){}
    virtual void build() = 0;
  protected:
    vars::Vars&vars;
};

}
