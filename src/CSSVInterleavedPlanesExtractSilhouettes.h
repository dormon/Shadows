#pragma once

#include<CSSVExtractSilhouettes.h>

class CSSVInterleavedPlanesExtractSilhouettes: public CSSVExtractSilhouettes{
  public:
    CSSVInterleavedPlanesExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};

