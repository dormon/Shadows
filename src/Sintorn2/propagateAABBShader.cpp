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
  //if(gl_LocalInvocationIndex == 0){
  //  float ab;
  //  ab = reductionArray[0];
  //  for(int i=1;i<64;++i)
  //    ab = max(ab,reductionArray[i]);
  //  reductionArray[0] = ab;
  //}
  //return;

  float ab[2];
  uint w;

  if(gl_LocalInvocationIndex < 32){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex + 32u];                 
    w = uint(ab[1]>ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 16){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex + 16u];                 
    w = uint(ab[1]>ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 8){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  8u];                 
    w = uint(ab[1]>ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 4){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  4u];                 
    w = uint(ab[1]>ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 2){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  2u];                 
    w = uint(ab[1]>ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 1){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  1u];                 
    w = uint(ab[1]>ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

}

void reduceMin(){
#line 52
  //if(gl_LocalInvocationIndex == 0){
  //  float ab;
  //  ab = reductionArray[0];
  //  for(int i=1;i<64;++i)
  //    ab = min(ab,reductionArray[i]);
  //  reductionArray[0] = ab;
  //}
  //return;

//*
  float ab[2];
  uint w;

  if(gl_LocalInvocationIndex < 32){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex + 32u];                 
    w = uint(ab[1]<ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 16){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex + 16u];                 
    w = uint(ab[1]<ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 8){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  8u];                 
    w = uint(ab[1]<ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 4){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  4u];                 
    w = uint(ab[1]<ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 2){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  2u];                 
    w = uint(ab[1]<ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

  if(gl_LocalInvocationIndex < 1){                         
    ab[0] = reductionArray[gl_LocalInvocationIndex +  0u];                 
    ab[1] = reductionArray[gl_LocalInvocationIndex +  1u];                 
    w = uint(ab[1]<ab[0]);
    reductionArray[gl_LocalInvocationIndex] = ab[w];
  }                                                                        

// */
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


  //if(gl_LocalInvocationIndex == 0){
  //  aabbPool[aabbLevelOffsetInFloats[destLevel]+node*6u+0u] = 1+(node+1)*100;//aabb[0];
  //  aabbPool[aabbLevelOffsetInFloats[destLevel]+node*6u+1u] = 2+(node+1)*100;//aabb[1];
  //  aabbPool[aabbLevelOffsetInFloats[destLevel]+node*6u+2u] = 3+(node+1)*100;//aabb[2];
  //  aabbPool[aabbLevelOffsetInFloats[destLevel]+node*6u+3u] = 4+(node+1)*100;//aabb[3];
  //  aabbPool[aabbLevelOffsetInFloats[destLevel]+node*6u+4u] = 5+(node+1)*100;//aabb[4];
  //  aabbPool[aabbLevelOffsetInFloats[destLevel]+node*6u+5u] = 6+(node+1)*100;//aabb[5];
  //}
  //return;

//*

  uint bit = node & warpMask;
  if(uint(nodePool[nodeLevelOffsetInUints[destLevel]+(node>>warpBits)*uintsPerWarp + uint(bit>31u)]&(1u<<(bit&0x1fu))) == 0u)
    return;

  uint isActive = uint(nodePool[nodeLevelOffsetInUints[destLevel+1u]+node*uintsPerWarp + uint(gl_LocalInvocationIndex>31)] & uint(1u<<(uint(gl_LocalInvocationIndex)&0x1fu)));

  uint64_t activeThreads = ballotARB(isActive != 0);
  uint selectedBit       = unpackUint2x32(activeThreads)[0]!=0u?findLSB(unpackUint2x32(activeThreads)[0]):findLSB(unpackUint2x32(activeThreads)[1])+32u;


#line 154
  float aabb[6] = {1337,1337,1337,1337,1337,1337};

  if(isActive != 0){
    aabb[0] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+0u];
    aabb[1] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+1u];
    aabb[2] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+2u];
    aabb[3] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+3u];
    aabb[4] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+4u];
    aabb[5] = aabbPool[aabbLevelOffsetInFloats[destLevel+1]+node*WARP*floatsPerAABB+uint(gl_LocalInvocationIndex)*floatsPerAABB+5u];
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
    //aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+0u] = (node+1)*100 + 1;//aabb[0];
    //aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+1u] = (node+1)*100 + 2;//aabb[1];
    //aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+2u] = (node+1)*100 + 3;//aabb[2];
    //aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+3u] = (node+1)*100 + 4;//aabb[3];
    //aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+4u] = (node+1)*100 + 5;//aabb[4];
    //aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+5u] = (node+1)*100 + 6;//aabb[5];

    aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+0u] = aabb[0];
    aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+1u] = aabb[1];
    aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+2u] = aabb[2];
    aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+3u] = aabb[3];
    aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+4u] = aabb[4];
    aabbPool[aabbLevelOffsetInFloats[destLevel]+node*floatsPerAABB+5u] = aabb[5];
  }

// */
#endif
}

).";
