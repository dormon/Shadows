#pragma once

#include<cstdint>
#include<vector>
#include<numeric>
#include<algorithm>

#include <requiredBits.h>
#include <divRoundUp.h>

namespace sintorn2{

inline std::vector<uint32_t>computeNofNodesPerLevel(uint32_t allBits,uint32_t warpBits){
  std::vector<uint32_t>nofNodesPerLevel;
  int32_t bits = allBits;
  while(bits>0){
    nofNodesPerLevel.push_back(1u<<bits);
    bits -= warpBits;
  }
  nofNodesPerLevel.push_back(1u);
  std::reverse(nofNodesPerLevel.begin(),nofNodesPerLevel.end());
  return nofNodesPerLevel;
}

inline std::vector<uint32_t>computeNodeLevelSizeInUints(std::vector<uint32_t>const&nofNodesPerLevel,uint32_t warp,uint32_t uintsPerWarp){
  std::vector<uint32_t>sizes;
  for(auto const&l:nofNodesPerLevel)
    sizes.push_back(divRoundUp(l,warp)*uintsPerWarp);
  return sizes;
}

template<typename T>
T sum(std::vector<T>const&v){
  return std::accumulate(v.begin(),v.end(),0);
}

inline std::vector<uint32_t>getOffsets(std::vector<uint32_t>const&sizes){
  std::vector<uint32_t>offsets(sizes.size()+1);
  offsets[0] = 0;
  std::partial_sum(sizes.begin(),sizes.end(), offsets.begin()+1,std::plus<uint32_t>());
  offsets.pop_back();
  return offsets;
}

class Config{
  public:
    Config(
        uint32_t wavefrontSize = 64u    ,
        uint32_t winX          = 512u   ,
        uint32_t winY          = 512u   ,
        uint32_t tX            = 8u     ,
        uint32_t tY            = 8u     ,
        uint32_t minZ          = 9u     ){
        //float    nn            = 0.01f  ,
        //float    ff            = 1000.f ,
        //float    fo            = 3.1415f){
      windowX          = winX;
      windowY          = winY;
      tileX            = tX;
      tileY            = tY;
      minZBits         = minZ;
      warpBits         = requiredBits(wavefrontSize);
      clustersX        = divRoundUp(windowX,tileX);
      clustersY        = divRoundUp(windowY,tileY);
      xBits            = requiredBits(clustersX);
      yBits            = requiredBits(clustersY);
      zBits            = minZBits>0?minZBits:glm::max(glm::max(xBits,yBits),minZBits);
      clustersZ        = 1 << zBits;
      allBits          = xBits + yBits + zBits;
      nofLevels        = divRoundUp(allBits,warpBits);
      uintsPerWarp     = wavefrontSize / (sizeof(uint32_t)*8);
      nofNodesPerLevel = computeNofNodesPerLevel(allBits,warpBits);
      nodeLevelSize    = computeNodeLevelSizeInUints(nofNodesPerLevel,wavefrontSize,uintsPerWarp);
      nodesSize        = sum(nodeLevelSize) * sizeof(uint32_t);
      nodeLevelOffset  = getOffsets(nofNodesPerLevel);
      aabbsSize        = sum(nofNodesPerLevel) * floatsPerAABB * sizeof(float);
      //nnear         = nn;
      //ffar          = ff;
      //fovy          = fo;
    }
    uint32_t windowX     ;
    uint32_t windowY     ;
    uint32_t tileX       ;
    uint32_t tileY       ;
    uint32_t minZBits    ;
    uint32_t warpBits    ;
    uint32_t clustersX   ;
    uint32_t clustersY   ;
    uint32_t xBits       ;
    uint32_t yBits       ;
    uint32_t zBits       ;
    uint32_t clustersZ   ;
    uint32_t allBits     ;
    uint32_t nofLevels   ;
    uint32_t uintsPerWarp;
    std::vector<uint32_t>nofNodesPerLevel;
    std::vector<uint32_t>nodeLevelSize   ;
    uint32_t             nodesSize       ;
    uint32_t             aabbsSize       ;
    std::vector<uint32_t>nodeLevelOffset ;
    uint32_t const floatsPerAABB = 6;
    //float    nnear       ;
    //float    ffar        ;
    //float    fovy        ;
};

}
