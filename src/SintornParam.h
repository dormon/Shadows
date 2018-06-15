#pragma once

#include<cstdint>
#include<memory>
#include<ArgumentViewer/Fwd.h>

struct SintornParams{
  std::size_t shadowFrustaPerWorkGroup = 1    ;
  float       bias                     = 0.01f;
  bool        discardBackFacing        = true ;
  std::size_t shadowFrustaWGS          = 64   ;
};

SintornParams loadSintornParams(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);
