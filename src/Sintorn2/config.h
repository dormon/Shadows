#pragma once

#include<cstdint>

#include <requiredBits.h>
#include <divRoundUp.h>

namespace sintorn2{

class Config{
  public:
    Config(
        uint32_t wavefrontSize = 64u    ,
        uint32_t windowX       = 512u   ,
        uint32_t windowY       = 512u   ,
        uint32_t tileX         = 8u     ,
        uint32_t tileY         = 8u     ,
        uint32_t minZBits      = 9u     ,
        float    nn            = 0.01f  ,
        float    ff            = 1000.f ,
        float    fo            = 3.1415f){
      warpBits      = requiredBits(wavefrontSize);
      clustersX     = divRoundUp(windowX,tileX);
      clustersY     = divRoundUp(windowY,tileY);
      xBits         = requiredBits(clustersX);
      yBits         = requiredBits(clustersY);
      zBits         = minZBits>0?minZBits:glm::max(glm::max(xBits,yBits),minZBits);
      clustersZ     = 1 << zBits;
      allBits       = xBits + yBits + zBits;
      nofLevels     = divRoundUp(allBits,warpBits);
      uintsPerWarp  = wavefrontSize / (sizeof(uint32_t)*8);
      nnear         = nn;
      ffar          = ff;
      fovy          = fo;
    }
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
    float    nnear       ;
    float    ffar        ;
    float    fovy        ;
};

}
