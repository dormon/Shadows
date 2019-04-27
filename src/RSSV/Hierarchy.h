#pragma once

#include<vector>
#include<glm/glm.hpp>

namespace rssv{
  class Hierarchy;
  void printHierarchy(Hierarchy const&h);
}

class rssv::Hierarchy{
  public:
    Hierarchy(glm::uvec2 const&windowSize,uint32_t branchingFactor);
    std::vector<glm::uvec2>levelSize;
    std::vector<glm::uvec2>tileCount;
    std::vector<glm::uvec2>fullTileSize;
    std::vector<glm::uvec2>fullTileSizeInPixels;
    std::vector<glm::uvec2>fullTileExponent;
    std::vector<glm::uvec2>fullTileMask;
    std::vector<glm::uvec2>fullTileExponentPrev;
    std::vector<glm::uvec2>fullTileMaskPrev;
    std::vector<glm::vec2 >fullTileSizeInClipSpace;
    std::vector<glm::uvec2>fullTileCount;
    std::vector<glm::uvec2>borderTileSize;
    std::vector<glm::uvec2>borderTileSizeInPixels;
    std::vector<glm::vec2 >borderTileSizeInClipSpace;
    uint32_t nofLevels;
};

