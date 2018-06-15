#pragma once

#include<cstdint>
#include<memory>
#include<glm/glm.hpp>
#include<ArgumentViewer/Fwd.h>

struct RSSVParams{
  std::size_t computeSilhouetteWGS         = 64             ;
  bool        localAtomic                  = true           ;
  bool        cullSides                    = false          ;
  std::size_t silhouettesPerWorkgroup      = 1              ;
  bool        usePlanes                    = 0              ;
  glm::uvec2  copyDepthToLastLevelOfHDTWGS = glm::uvec2(8,8);
};

RSSVParams loadRSSVParams(std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);
