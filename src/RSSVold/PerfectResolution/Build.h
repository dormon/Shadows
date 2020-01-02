#pragma once

#include<Vars/Vars.h>

#include<RSSV/BuildHierarchy.h>

namespace rssv{

class PerfectHierarchy: public BuildHierarchy{
  public:
    PerfectHierarchy(vars::Vars&vars);
    virtual void build() override;
  protected:
};

}
