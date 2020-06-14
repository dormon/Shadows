#include <RSSV/mergeShader.h>

std::string const rssv::mergeShaderFWD = R".(
void mergeJOB();
).";

std::string const rssv::mergeShader = R".(

uint getMorton(uvec2 coord,float depth){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float z = DEPTH_TO_Z(depth);
  uint  zQ = QUANTIZE_Z(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}

void merge(uint job){
  uvec2 tile       = uvec2(job%clustersX,job/clustersX);
  uvec2 localCoord = uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);
  uvec2 coord      = tile*uvec2(TILE_X,TILE_Y) + localCoord;
  if(any(greaterThanEqual(coord,uvec2(WINDOW_X,WINDOW_Y))))return;
  int mult = imageLoad(stencil,ivec2(coord)).x;

  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  uint morton = getMorton(coord,depth);

  if(nofLevels>0){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-1u,0u,5u)]+(morton>>(warpBits*0))];
  }
  if(nofLevels>1){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-2u,0u,5u)]+(morton>>(warpBits*1))];
  }
  if(nofLevels>2){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-3u,0u,5u)]+(morton>>(warpBits*2))];
  }
  if(nofLevels>3){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-4u,0u,5u)]+(morton>>(warpBits*3))];
  }
  if(nofLevels>4){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-5u,0u,5u)]+(morton>>(warpBits*4))];
  }
  if(nofLevels>5){
    mult += bridges[nodeLevelOffset[clamp(nofLevels-6u,0u,5u)]+(morton>>(warpBits*5))];
  }

  if(mult != 0)
    imageStore(shadowMask,ivec2(coord),vec4(0.f));
}

void mergeJOB(){
#if PERFORM_MERGE == 1
  //every WGS increments counter
  if(gl_LocalInvocationIndex == 0){
    atomicAdd(traverseDoneCounter,1);
  }
  //barrier();

  //and wait until all WGS finish
  for(int i=0;i<1000;++i){
    uint finishedWGS;
    int canWeContinue = 0;
    if(gl_LocalInvocationIndex == 0){
      finishedWGS = traverseDoneCounter;
      if(finishedWGS >= gl_NumWorkGroups.x)
        canWeContinue = 1;
    }
    canWeContinue = readFirstInvocationARB(canWeContinue);
    //barrier();
    if(canWeContinue == 1){
      //if(gl_LocalInvocationIndex==0)
      //  atomicAdd(dummy[0],1);
      //barrier();
      //return;
      break;
    }

    if(gl_LocalInvocationIndex == 0){
      uint c=0;
      for(uint j=0;j<100;++j)
        c = (c+finishedWGS+j)%177;
      if(c == 1337)dummy[2] = 1111;
    }
    //barrier();
  }

  //if(gl_LocalInvocationIndex==0)
  //  atomicAdd(dummy[1],1);
  //dummy[4] = gl_NumWorkGroups.x;
  
  //return;
  //todo wait for finish



  uint job = 0u;
  //triangle loop
  for(;;){
  
    if(gl_LocalInvocationIndex==0)
      job = atomicAdd(mergeJobCounter,1);

    job = readFirstInvocationARB(job);
    if(job >= clustersX*clustersY)break;

    merge(job);
  }
#endif
}
).";

#if 0
std::string const rssv::mergeShader = R".(
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

  float z = DEPTH_TO_Z(depth);
  uint  zQ = QUANTIZE_Z(z);
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

#endif
