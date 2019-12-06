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


uniform uint destLevel = 0;


#if WARP == 64

shared float reductionArray[WARP*6u];

void reduce(){
  float ab[2];
  uint w;

  //6*64 -> 6*32
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&32u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  
  ab[0] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&32u)!=0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[WARP*4u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*4u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&32u)!=0u)) > 0.f);
  reductionArray[WARP*2u + gl_LocalInvocationIndex] = ab[w];
  
  //6*32 -> 6*16
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&16u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  
  ab[0] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*2u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&16u)!=0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];

  //6*16 -> 6*8
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&8u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  //6*8 -> 6*4
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&4u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  //6*4 -> 6*2
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&2u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  //6*2 -> 6*1
  ab[0] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u + (uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&1u)!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
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
  const uint floatsPerAABB   = 6u;

  const uint warpMask        = uint(WARP - 1u);

  const uint nodesPerLevel[6] = {
    1u << uint(max(int(allBits) - int((nofLevels-1u)*warpBits),0)),
    1u << uint(max(int(allBits) - int((nofLevels-2u)*warpBits),0)),
    1u << uint(max(int(allBits) - int((nofLevels-3u)*warpBits),0)),
    1u << uint(max(int(allBits) - int((nofLevels-4u)*warpBits),0)),
    1u << uint(max(int(allBits) - int((nofLevels-5u)*warpBits),0)),
    1u << uint(max(int(allBits) - int((nofLevels-6u)*warpBits),0)),
  };

  const uint nodeLevelSizeInUints[6] = {
    (nodesPerLevel[0] >> warpBits) * uintsPerWarp,
    (nodesPerLevel[1] >> warpBits) * uintsPerWarp,
    (nodesPerLevel[2] >> warpBits) * uintsPerWarp,
    (nodesPerLevel[3] >> warpBits) * uintsPerWarp,
    (nodesPerLevel[4] >> warpBits) * uintsPerWarp,
    (nodesPerLevel[5] >> warpBits) * uintsPerWarp,
  };

  const uint nodeLevelOffsetInUints[6] = {
    0,
    0 + nodeLevelSizeInUints[0],
    0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1],
    0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1] + nodeLevelSizeInUints[2],
    0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1] + nodeLevelSizeInUints[2] + nodeLevelSizeInUints[3],
    0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1] + nodeLevelSizeInUints[2] + nodeLevelSizeInUints[3] + nodeLevelSizeInUints[4],
  };

  const uint aabbLevelSizeInFloats[6] = {
    nodesPerLevel[0] * floatsPerAABB,
    nodesPerLevel[1] * floatsPerAABB,
    nodesPerLevel[2] * floatsPerAABB,
    nodesPerLevel[3] * floatsPerAABB,
    nodesPerLevel[4] * floatsPerAABB,
    nodesPerLevel[5] * floatsPerAABB,
  };

  const uint aabbLevelOffsetInFloats[6] = {
    0,
    0 + aabbLevelSizeInFloats[0],
    0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1],
    0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2],
    0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2] + aabbLevelSizeInFloats[3],
    0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2] + aabbLevelSizeInFloats[3] + aabbLevelSizeInFloats[4],
  };





#line 141
#if WARP == 64
  uint node = gl_WorkGroupID.x + gl_WorkGroupID.y*gl_NumWorkGroups.x;

//*

  uint bit = node & warpMask;
  if(uint(nodePool[nodeLevelOffsetInUints[destLevel]+(node>>warpBits)*uintsPerWarp + uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u)
    return;

  uint isActive = uint(nodePool[nodeLevelOffsetInUints[destLevel+1u]+node*uintsPerWarp + uint(gl_LocalInvocationIndex>31)] & uint(1u<<(uint(gl_LocalInvocationIndex)&0x1fu)));

  uint64_t activeThreads = ballotARB(isActive != 0);
  uint selectedBit       = unpackUint2x32(activeThreads)[0]!=0u?findLSB(unpackUint2x32(activeThreads)[0]):findLSB(unpackUint2x32(activeThreads)[1])+32u;


#line 154
  if(isActive != 0){
    reductionArray[WARP*0u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+0u];
    reductionArray[WARP*1u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+1u];
    reductionArray[WARP*2u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+2u];
    reductionArray[WARP*3u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+3u];
    reductionArray[WARP*4u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+4u];
    reductionArray[WARP*5u+uint(gl_LocalInvocationIndex)] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+5u];
  }

  if(isActive == 0){
    reductionArray[WARP*0u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*0u+selectedBit];
    reductionArray[WARP*1u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*1u+selectedBit];
    reductionArray[WARP*2u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*2u+selectedBit];
    reductionArray[WARP*3u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*3u+selectedBit];
    reductionArray[WARP*4u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*4u+selectedBit];
    reductionArray[WARP*5u+uint(gl_LocalInvocationIndex)] = reductionArray[WARP*5u+selectedBit];
  }

  reduce();

  if(gl_LocalInvocationIndex < floatsPerAABB)
    aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+gl_LocalInvocationIndex] = reductionArray[gl_LocalInvocationIndex];

// */
#endif
}

).";
