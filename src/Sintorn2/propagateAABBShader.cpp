#include <Sintorn2/propagateAABBShader.h>

std::string const sintorn2::propagateAABBShader = R".(

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

layout(local_size_x=WARP)in;

layout(binding=0)buffer NodePool   {uint  nodePool   [];};
layout(binding=1)buffer AABBPool   {float aabbPool   [];};

shared float reductionArray[WARP];

uniform uint destLevel = 0;

#if WARP == 64

void reduceMax(){
  if(gl_LocalInvocationIndex == 0){
    float ab;
    ab = reductionArray[0];
    for(int i=1;i<64;++i)
      ab = max(ab,reductionArray[i]);
    reductionArray[0] = ab;
  }
  return;
}

void reduceMin(){
#line 52
  if(gl_LocalInvocationIndex == 0){
    float ab;
    ab = reductionArray[0];
    for(int i=1;i<64;++i)
      ab = min(ab,reductionArray[i]);
    reductionArray[0] = ab;
  }
  return;

/*
  float ab[2];
  uint w;


  ab[0] = reductionArray[(uint(gl_LocalInvocationIndex)&0x1fu)+ 0u];       
  ab[1] = reductionArray[(uint(gl_LocalInvocationIndex)&0x1fu)+32u];       
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&0x20u)!=0)) > 0.f);
  reductionArray[gl_LocalInvocationIndex] = ab[w];                         


  if((uint(gl_LocalInvocationIndex)&0x10u) == 0u){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex + 16u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&0x20u)!=0u)) > 0.f);
    reductionArray[gl_LocalInvocationIndex - uint(uint((gl_LocalInvocationIndex)&0x20u) != 0u)*16u] = ab[w];
  }                                                                        
                                                                           
  if((uint(gl_LocalInvocationIndex)&0x28u) == 0u){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex + 0u];                  
    ab[1] = reductionArray[gl_LocalInvocationIndex + 8u];                  
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&0x10u)!=0u)) > 0.f);
    reductionArray[gl_LocalInvocationIndex - uint(uint((gl_LocalInvocationIndex)&0x10u) != 0u)*8u] = ab[w];
  }                                                                        
                                                                           
  if((uint(gl_LocalInvocationIndex)&0x34u) == 0u){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex + 0u];                  
    ab[1] = reductionArray[gl_LocalInvocationIndex + 4u];                  
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&0x8u)!=0u)) > 0.f);
    reductionArray[gl_LocalInvocationIndex - uint(uint((gl_LocalInvocationIndex)&0x8u) != 0u)*4u] = ab[w];
  }                                                                        
                                                                           
  if((uint(gl_LocalInvocationIndex)&0x3au) == 0u){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex + 0u];                  
    ab[1] = reductionArray[gl_LocalInvocationIndex + 2u];                  
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&0x4u)!=0u)) > 0.f);
    reductionArray[gl_LocalInvocationIndex - uint(uint((gl_LocalInvocationIndex)&0x4u) != 0u)*2u] = ab[w];
  }                                                                        
                                                                           
  if((uint(gl_LocalInvocationIndex)&0x3du) == 0u){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex + 0u];                  
    ab[1] = reductionArray[gl_LocalInvocationIndex + 1u];                  
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&0x2u)!=0u)) > 0.f);
    reductionArray[gl_LocalInvocationIndex - uint(uint((gl_LocalInvocationIndex)&0x2u) != 0u)*1u] = ab[w];
  }
*/
}
#endif
#line 110
void main(){
  const uint warpBits        = uint(ceil(log2(float(WARP))));
  const uint clustersX       = uint(WINDOW_X/TILE_X) + uint(WINDOW_X%TILE_X != 0u);
  const uint clustersY       = uint(WINDOW_Y/TILE_Y) + uint(WINDOW_Y%TILE_Y != 0u);
  const uint xBits           = uint(ceil(log2(float(clustersX))));
  const uint yBits           = uint(ceil(log2(float(clustersY))));
  const uint zBits           = MIN_Z_BITS>0?MIN_Z_BITS:max(max(xBits,yBits),MIN_Z_BITS);
  const uint allBits         = xBits + yBits + zBits;
  const uint nofLevels       = uint(allBits/warpBits) + uint(allBits%warpBits != 0u);
  const uint uintsPerWarp    = uint(WARP/32u);

  const uint warpMask        = uint(WARP - 1u);

  const uint levelSize[6] = {
    uintsPerWarp << uint(max(int(allBits) - int((nofLevels-0u)*warpBits),0)),
    uintsPerWarp << uint(max(int(allBits) - int((nofLevels-1u)*warpBits),0)),
    uintsPerWarp << uint(max(int(allBits) - int((nofLevels-2u)*warpBits),0)),
    uintsPerWarp << uint(max(int(allBits) - int((nofLevels-3u)*warpBits),0)),
    uintsPerWarp << uint(max(int(allBits) - int((nofLevels-4u)*warpBits),0)),
    uintsPerWarp << uint(max(int(allBits) - int((nofLevels-5u)*warpBits),0)),
  };

  const uint levelOffset[6] = {
    0,
    0 + levelSize[0],
    0 + levelSize[0] + levelSize[1],
    0 + levelSize[0] + levelSize[1] + levelSize[2],
    0 + levelSize[0] + levelSize[1] + levelSize[2] + levelSize[3],
    0 + levelSize[0] + levelSize[1] + levelSize[2] + levelSize[3] + levelSize[4],
  };
#line 141
#if WARP == 64
  uint node = gl_WorkGroupID.x;
  uint bit = node & 0x3fu;
  if(uint(nodePool[levelOffset[destLevel]+(node>>6u)*2u + uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u)
    return;

  uint isActive = uint(nodePool[levelOffset[destLevel+1u]+(node&0xffffffc0u) + uint(gl_LocalInvocationIndex>31)] & uint(1u<<(uint(gl_LocalInvocationIndex)&0x1fu)));

  uint64_t activeThreads = ballotARB(isActive != 0);
  uint selectedBit       = unpackUint2x32(activeThreads)[0]!=0u?findLSB(unpackUint2x32(activeThreads)[0]):findLSB(unpackUint2x32(activeThreads)[1])+32u;


#line 154
  float aabb[6];

  if(isActive != 0){
    aabb[0] = aabbPool[levelOffset[destLevel+1]*3u*64u+(node&0xffffffc0u)*6u+uint(gl_LocalInvocationIndex)*6u+0u];
    aabb[1] = aabbPool[levelOffset[destLevel+1]*3u*64u+(node&0xffffffc0u)*6u+uint(gl_LocalInvocationIndex)*6u+1u];
    aabb[2] = aabbPool[levelOffset[destLevel+1]*3u*64u+(node&0xffffffc0u)*6u+uint(gl_LocalInvocationIndex)*6u+2u];
    aabb[3] = aabbPool[levelOffset[destLevel+1]*3u*64u+(node&0xffffffc0u)*6u+uint(gl_LocalInvocationIndex)*6u+3u];
    aabb[4] = aabbPool[levelOffset[destLevel+1]*3u*64u+(node&0xffffffc0u)*6u+uint(gl_LocalInvocationIndex)*6u+4u];
    aabb[5] = aabbPool[levelOffset[destLevel+1]*3u*64u+(node&0xffffffc0u)*6u+uint(gl_LocalInvocationIndex)*6u+5u];
  }

  if(isActive == 0){
    aabb[0] = readInvocationARB(aabb[0],selectedBit);
    aabb[1] = readInvocationARB(aabb[1],selectedBit);
    aabb[2] = readInvocationARB(aabb[2],selectedBit);
    aabb[3] = readInvocationARB(aabb[3],selectedBit);
    aabb[4] = readInvocationARB(aabb[4],selectedBit);
    aabb[5] = readInvocationARB(aabb[5],selectedBit);
  }

  reductionArray[gl_LocalInvocationIndex] = aabb[0];
  reduceMin();
  if(gl_LocalInvocationIndex == 0)
    aabb[0] = reductionArray[0];

  reductionArray[gl_LocalInvocationIndex] = aabb[1];
  reduceMax();
  if(gl_LocalInvocationIndex == 0)
    aabb[1] = reductionArray[0];

  reductionArray[gl_LocalInvocationIndex] = aabb[2];
  reduceMin();
  if(gl_LocalInvocationIndex == 0)
    aabb[2] = reductionArray[0];

  reductionArray[gl_LocalInvocationIndex] = aabb[3];
  reduceMax();
  if(gl_LocalInvocationIndex == 0)
    aabb[3] = reductionArray[0];

  reductionArray[gl_LocalInvocationIndex] = aabb[4];
  reduceMin();
  if(gl_LocalInvocationIndex == 0)
    aabb[4] = reductionArray[0];

  reductionArray[gl_LocalInvocationIndex] = aabb[5];
  reduceMax();
  if(gl_LocalInvocationIndex == 0)
    aabb[5] = reductionArray[0];


  if(gl_LocalInvocationIndex == 0){
    aabbPool[levelOffset[destLevel]*3u*64u+node*6u+0u] = aabb[0];
    aabbPool[levelOffset[destLevel]*3u*64u+node*6u+1u] = aabb[1];
    aabbPool[levelOffset[destLevel]*3u*64u+node*6u+2u] = aabb[2];
    aabbPool[levelOffset[destLevel]*3u*64u+node*6u+3u] = aabb[3];
    aabbPool[levelOffset[destLevel]*3u*64u+node*6u+4u] = aabb[4];
    aabbPool[levelOffset[destLevel]*3u*64u+node*6u+5u] = aabb[5];
  }

#endif

}

).";
