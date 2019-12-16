#include <Sintorn2/mergeShader.h>

std::string const sintorn2::mergeShader = R".(
#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef WINDOW_X
#define WINDOW_X 512
#endif//WINDOW_X

#ifndef WINDOW_Y
#define WINDOW_Y 512
#endif//WINDOW_Y

#ifndef TILE_X
#define TILE_X 8
#endif//TILE_X

#ifndef TILE_Y
#define TILE_Y 8
#endif//TILE_Y

#ifndef MIN_Z_BITS
#define MIN_Z_BITS 9
#endif//MIN_Z_BITS

#ifndef NEAR
#define NEAR 0.01f
#endif//NEAR

#ifndef FAR
#define FAR 1000.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer NodePool        {uint  nodePool        [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;

uint getMorton(uvec2 coord,float depth){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float z = depthToZ(depth);
  uint  zQ = quantizeZ(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}



void compute(uvec2 coord){


  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  uint morton = getMorton(coord,depth);

#line 120
  //if(uintsPerWarp == 1){
  #if WARP == 32
    if(nofLevels>0){
      uint bit  = (morton >> (warpBits*0u)) & warpMask;
      uint node = (morton >> (warpBits*1u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-1u,0u,5u)]+node]&(1u<<(bit))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>1){
      uint bit  = (morton >> (warpBits*1u)) & warpMask;
      uint node = (morton >> (warpBits*2u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-2u,0u,5u)]+node]&(1u<<(bit))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>2){
      uint bit  = (morton >> (warpBits*2u)) & warpMask;
      uint node = (morton >> (warpBits*3u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-3u,0u,5u)]+node]&(1u<<(bit))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>3){
      uint bit  = (morton >> (warpBits*3u)) & warpMask;
      uint node = (morton >> (warpBits*4u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-4u,0u,5u)]+node]&(1u<<(bit))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>4){
      uint bit  = (morton >> (warpBits*4u)) & warpMask;
      uint node = (morton >> (warpBits*5u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-5u,0u,5u)]+node]&(1u<<(bit))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>5){
      uint bit  = (morton >> (warpBits*5u)) & warpMask;
      uint node = (morton >> (warpBits*6u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-6u,0u,5u)]+node]&(1u<<(bit))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }
  #endif
  //}
 

#line 166
  #if WARP == 64

    if(nofLevels>0){
      uint bit  = (morton >> (warpBits*0u)) & warpMask;
      uint node = (morton >> (warpBits*1u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-1u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>1){
      uint bit  = (morton >> (warpBits*1u)) & warpMask;
      uint node = (morton >> (warpBits*2u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-2u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>2){
      uint bit  = (morton >> (warpBits*2u)) & warpMask;
      uint node = (morton >> (warpBits*3u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-3u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>3){
      uint bit  = (morton >> (warpBits*3u)) & warpMask;
      uint node = (morton >> (warpBits*4u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-4u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>4){
      uint bit  = (morton >> (warpBits*4u)) & warpMask;
      uint node = (morton >> (warpBits*5u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-5u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }

    if(nofLevels>5){
      uint bit  = (morton >> (warpBits*5u)) & warpMask;
      uint node = (morton >> (warpBits*6u));
      if(uint(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-6u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u){
        imageStore(shadowMask,ivec2(coord),vec4(0));
        return;
      }
    }


  #endif
}


void main(){
  const uint loCoordShift  = uint(ceil(log2(float(TILE_X))));
  const uint loCoordMask   = uint(TILE_X-1u);

  uvec2 loCoord = uvec2(uint(gl_LocalInvocationIndex)&loCoordMask,uint(gl_LocalInvocationIndex)>>loCoordShift);
  uvec2 wgCoord = uvec2(gl_WorkGroupID.xy) * uvec2(TILE_X,TILE_Y);

#if WARP == 64
  uvec2 coord = wgCoord + loCoord;
  if(any(greaterThanEqual(coord,uvec2(WINDOW_X,WINDOW_Y))))
    return;
  compute(coord);
#else
  uvec2 coord = wgCoord + loCoord;
  if(any(greaterThanEqual(coord,uvec2(WINDOW_X,WINDOW_Y))))
    return;
  compute(coord);
  coord += uvec2(0u,4u);
  if(any(greaterThanEqual(coord,uvec2(WINDOW_X,WINDOW_Y))))
    return;
  compute(coord);
#endif
}

).";
