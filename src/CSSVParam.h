#pragma once

#include<cstdint>
#include<memory>
#include<ArgumentViewer/Fwd.h>

struct CSSVParams{
  std::size_t computeSidesWGS = 64   ;
  bool   localAtomic          = true ;
  bool   cullSides            = false;
  bool   usePlanes            = false;
  bool   useInterleaving      = false;
};

CSSVParams loadCSSVParams(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);

