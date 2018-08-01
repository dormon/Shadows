#pragma once

#include<CSSV/ExtractSilhouettes.h>

class cssv::PlanesExtractSilhouettes: public ExtractSilhouettes{
  public:
    PlanesExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};

