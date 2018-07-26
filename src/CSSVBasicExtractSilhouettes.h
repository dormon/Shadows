#pragma once

#include<CSSVExtractSilhouettes.h>

class CSSVBasicExtractSilhouettes: public CSSVExtractSilhouettes{
  public:
    CSSVBasicExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};

