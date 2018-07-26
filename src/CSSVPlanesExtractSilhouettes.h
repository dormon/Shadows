#pragma once

#include<CSSVExtractSilhouettes.h>

class CSSVPlanesExtractSilhouettes: public CSSVExtractSilhouettes{
  public:
    CSSVPlanesExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};

