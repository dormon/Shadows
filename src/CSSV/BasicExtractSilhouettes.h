#pragma once

#include<CSSV/ExtractSilhouettes.h>

class cssv::BasicExtractSilhouettes: public ExtractSilhouettes{
  public:
    BasicExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};

