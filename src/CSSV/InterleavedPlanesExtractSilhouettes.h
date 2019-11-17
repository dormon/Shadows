#pragma once

#include<CSSV/ExtractSilhouettes.h>

class cssv::InterleavedPlanesExtractSilhouettes: public ExtractSilhouettes{
  public:
    InterleavedPlanesExtractSilhouettes(vars::Vars&vars);
    //virtual void compute(glm::vec4 const&lightPosition) override;
};

