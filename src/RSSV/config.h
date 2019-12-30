#pragma once

#include<iostream>
#include<cstdint>
#include<vector>
#include<numeric>
#include<algorithm>

#include <requiredBits.h>
#include <divRoundUp.h>

namespace rssv{

inline std::vector<uint32_t>computeNofNodesPerLevel(uint32_t allBits,uint32_t warpBits){
  std::vector<uint32_t>nofNodesPerLevel;
  int32_t bits = allBits;
  while(bits>0){
    nofNodesPerLevel.push_back(1u<<bits);
    bits -= warpBits;
  }
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

template<typename T>
std::vector<T> mul(std::vector<T>const&v,T const&c){
  std::vector<T>r;
  for(auto const&x:v)
    r.push_back(x*c);
  return r;
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
      windowX                 = winX;
      windowY                 = winY;
      tileX                   = tX;
      tileY                   = tY;
      minZBits                = minZ;
      warpBits                = uint32_t(requiredBits(wavefrontSize));
      clustersX               = divRoundUp(windowX,tileX);
      clustersY               = divRoundUp(windowY,tileY);
      xBits                   = uint32_t(requiredBits(clustersX));
      yBits                   = uint32_t(requiredBits(clustersY));
      zBits                   = minZBits>0?minZBits:glm::max(glm::max(xBits,yBits),minZBits);
      clustersZ               = 1 << zBits;
      allBits                 = xBits + yBits + zBits;
      nofLevels               = divRoundUp(allBits,warpBits);
      uintsPerWarp            = wavefrontSize / (sizeof(uint32_t)*8);
      nofNodesPerLevel        = computeNofNodesPerLevel(allBits,warpBits);
      nodeLevelSizeInUints    = computeNodeLevelSizeInUints(nofNodesPerLevel,wavefrontSize,uintsPerWarp);
      nodesSize               = sum(nodeLevelSizeInUints) * sizeof(uint32_t);
      nodeLevelOffsetInUints  = getOffsets(nodeLevelSizeInUints);
      nofNodes                = sum(nofNodesPerLevel);
      aabbsSize               = nofNodes * floatsPerAABB * sizeof(float);
      aabbLevelSizeInFloats   = mul(nofNodesPerLevel,floatsPerAABB);
      aabbLevelOffsetInFloats = getOffsets(aabbLevelSizeInFloats);
      nodeLevelOffset         = getOffsets(nofNodesPerLevel);
    }
    void print(){
#define PRINT(x) std::cerr << #x << ": " << x << std::endl
      PRINT(windowX     );
      PRINT(windowY     );
      PRINT(tileX       );
      PRINT(tileY       );
      PRINT(minZBits    );
      PRINT(warpBits    );
      PRINT(clustersX   );
      PRINT(clustersY   );
      PRINT(xBits       );
      PRINT(yBits       );
      PRINT(zBits       );
      PRINT(clustersZ   );
      PRINT(allBits     );
      PRINT(nofLevels   );
      PRINT(uintsPerWarp);
      PRINT(nodesSize   );
      PRINT(aabbsSize   );
      PRINT(nofNodes    );

      PRINT(floatsPerAABB);
#undef PRINT
#define PRINT(x) std::cerr << #x << ":" << std::endl;\
  for(auto const&a:x)\
    std::cerr << "  " << a << std::endl
    PRINT(nofNodesPerLevel        );
    PRINT(nodeLevelOffset         );
    PRINT(nodeLevelSizeInUints    );
    PRINT(nodeLevelOffsetInUints  );
    PRINT(aabbLevelSizeInFloats   );
    PRINT(aabbLevelOffsetInFloats );
#undef PRINT

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
    uint32_t nodesSize   ;
    uint32_t aabbsSize   ;
    uint32_t nofNodes    ;
    uint32_t const floatsPerAABB = 6;
    std::vector<uint32_t>nofNodesPerLevel        ;
    std::vector<uint32_t>nodeLevelOffset         ;
    std::vector<uint32_t>nodeLevelSizeInUints    ;
    std::vector<uint32_t>nodeLevelOffsetInUints  ;
    std::vector<uint32_t>aabbLevelSizeInFloats   ;
    std::vector<uint32_t>aabbLevelOffsetInFloats ;
};

}
