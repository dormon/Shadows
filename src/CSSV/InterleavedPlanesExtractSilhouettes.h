#pragma once

#include<CSSV/ExtractSilhouettes.h>

class cssv::InterleavedPlanesExtractSilhouettes: public ExtractSilhouettes{
  public:
    InterleavedPlanesExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
    virtual void compute(glm::vec4 const&lightPosition) override;
};

