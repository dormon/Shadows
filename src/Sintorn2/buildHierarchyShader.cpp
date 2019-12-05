#include <Sintorn2/buildHierarchyShader.h>

std::string const sintorn2::buildHierarchyShader = R".(
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

layout(binding=0)buffer NodePool   {uint  nodePool   [];};
layout(binding=1)buffer AABBPool   {float aabbPool   [];};

layout(binding=1)uniform sampler2DRect depthTexture;

uint getMorton(uvec2 coord,float depth){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float z = depthToZ(depth);
  uint  zQ = quantizeZ(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}

uint activeThread = 0;

#define USE_READ_INVOCATION

#ifndef USE_READ_INVOCATION
shared uint sharedMortons[WARP];
#endif

shared float reductionArray[WARP];

#if WARP == 64
void reduce(){
#line 67
  //if(gl_LocalInvocationIndex == 0){
  //  float ab[2];
  //  ab[0] = reductionArray[0];
  //  ab[1] = reductionArray[0];
  //  for(int i=1;i<64;++i){
  //    ab[0] = min(ab[0],reductionArray[i]);
  //    ab[1] = max(ab[1],reductionArray[i]);
  //  }
  //  reductionArray[0] = ab[0];
  //  reductionArray[1] = ab[1];
  //}
  //return;

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

}
#endif

void compute(uvec2 coord){
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
  const uint floatsPerAABB   = 6u;

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


  /*
  if(gl_GlobalInvocationID.x == 0){
    uint sx=300;
    uint sy=300;
    for(uint x=sx;x<512;++x)
      for(uint y=sx;y<512;++y){

        float depth = texelFetch(depthTexture,ivec2(x,y)).x*2.f-1.f;
        float z = depthToZ(depth);
        //nodePool[(y-sy)*(512-sx)+(x-sx)] = floatBitsToUint(z);
        uint  zQ = quantizeZ(z);
        nodePool[(y-sy)*(512-sx)+(x-sx)] = zQ;
      }
  }
  return;
  */

  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  uint morton = getMorton(coord,depth);
#ifndef USE_READ_INVOCATION
  sharedMortons[gl_LocalInvocationIndex] = morton;
#endif
#line 120
  //if(uintsPerWarp == 1){
  #if WARP == 32
    uint notDone = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread != 0)),0);
  
    uint counter = 0;
    while(notDone != 0){
      if(counter >= 32)break;
      counter ++;

      uint selectedBit     = findLSB(notDone);
      uint referenceMorton = sharedMortons[selectedBit];

      uint sameCluster = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton)),0);
      if(gl_LocalInvocationIndex == 0){

        //if(nofLevels>0)atomicOr(nodePool[(referenceMorton>>(warpBits*1))],1u<<((referenceMorton>>(warpBits*0))&warpMask));
        if(nofLevels>0)atomicOr(nodePool[levelOffset[clamp(nofLevels-1u,0u,5u)]+(referenceMorton>>(warpBits*1))],1u<<((referenceMorton>>(warpBits*0))&warpMask));
        if(nofLevels>1)atomicOr(nodePool[levelOffset[clamp(nofLevels-2u,0u,5u)]+(referenceMorton>>(warpBits*2))],1u<<((referenceMorton>>(warpBits*1))&warpMask));
        if(nofLevels>2)atomicOr(nodePool[levelOffset[clamp(nofLevels-3u,0u,5u)]+(referenceMorton>>(warpBits*3))],1u<<((referenceMorton>>(warpBits*2))&warpMask));
        if(nofLevels>3)atomicOr(nodePool[levelOffset[clamp(nofLevels-4u,0u,5u)]+(referenceMorton>>(warpBits*4))],1u<<((referenceMorton>>(warpBits*3))&warpMask));
        if(nofLevels>4)atomicOr(nodePool[levelOffset[clamp(nofLevels-5u,0u,5u)]+(referenceMorton>>(warpBits*5))],1u<<((referenceMorton>>(warpBits*4))&warpMask));
        if(nofLevels>5)atomicOr(nodePool[levelOffset[clamp(nofLevels-6u,0u,5u)]+(referenceMorton>>(warpBits*6))],1u<<((referenceMorton>>(warpBits*5))&warpMask));
      }
      notDone ^= sameCluster;
    }
  #endif
  //}
 
#line 166
  //if(uintsPerWarp == 2){
  #if WARP == 64
    //if(gl_GlobalInvocationID.x == 0){
    //  nodePool[0] = levelSize[0];
    //  nodePool[1] = levelSize[1];
    //  nodePool[2] = levelSize[2];
    //  nodePool[3] = levelSize[3];
    //  nodePool[4] = levelSize[4];
    //  nodePool[5] = levelSize[5];
    //}
    //return;
    uint counter = 0;
    uint64_t notDone = ballotARB(activeThread != 0);
    while(notDone != 0){

      if(counter >= 64)break;
      counter++;

      uint selectedBit     = unpackUint2x32(notDone)[0]!=0?findLSB(unpackUint2x32(notDone)[0]):findLSB(unpackUint2x32(notDone)[1])+32u;
      //uint selectedBit     = (notDone&0xfffffffful)!=0?findLSB(uint(notDone)):findLSB(uint(notDone>>32u))+32u;
      
#ifndef USE_READ_INVOCATION
      uint referenceMorton = sharedMortons[selectedBit];
#else
      uint referenceMorton = readInvocationARB(morton,selectedBit);
#endif

      if(gl_LocalInvocationIndex == 0){
        if(nofLevels>0){
          uint bit  = (referenceMorton >> (warpBits*0u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*1u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-1u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>1){
          uint bit  = (referenceMorton >> (warpBits*1u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*2u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-2u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>2){
          uint bit  = (referenceMorton >> (warpBits*2u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*3u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-3u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>3){
          uint bit  = (referenceMorton >> (warpBits*3u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*4u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-4u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>4){
          uint bit  = (referenceMorton >> (warpBits*4u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*5u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-5u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>5){
          uint bit  = (referenceMorton >> (warpBits*5u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*6u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-6u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
      }
      
      uint64_t sameCluster = ballotARB(referenceMorton == morton && activeThread != 0);



      ///XXX
      reductionArray[gl_LocalInvocationIndex] = -1.f + 2.f/float(WINDOW_X)*(coord.x+0.5f);

      if(referenceMorton != morton || activeThread == 0)
        reductionArray[gl_LocalInvocationIndex] = reductionArray[selectedBit];

      reduce();

      if(gl_LocalInvocationIndex == 0){
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+0] = reductionArray[0];
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+1] = reductionArray[1];
      }

      ///YYY
      reductionArray[gl_LocalInvocationIndex] = -1.f + 2.f/float(WINDOW_Y)*(coord.y+0.5f);

      if(referenceMorton != morton || activeThread == 0)
        reductionArray[gl_LocalInvocationIndex] = reductionArray[selectedBit];

      reduce();

      if(gl_LocalInvocationIndex == 0){
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+2] = reductionArray[0];
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+3] = reductionArray[1];
      }

      ///ZZZ
      reductionArray[gl_LocalInvocationIndex] = depth;

      if(referenceMorton != morton || activeThread == 0)
        reductionArray[gl_LocalInvocationIndex] = reductionArray[selectedBit];

      reduce();

      if(gl_LocalInvocationIndex == 0){
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+4] = reductionArray[0];
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+5] = reductionArray[1];
      }



      notDone ^= sameCluster;
    }
  #endif
  //}
}


void main(){
  const uint loCoordShift  = uint(ceil(log2(float(TILE_X))));
  const uint loCoordMask   = uint(TILE_X-1u);

  uvec2 loCoord = uvec2(uint(gl_LocalInvocationIndex)&loCoordMask,uint(gl_LocalInvocationIndex)>>loCoordShift);
  uvec2 wgCoord = uvec2(gl_WorkGroupID.xy) * uvec2(TILE_X,TILE_Y);
  uvec2 coord = wgCoord + loCoord;
  activeThread = uint(all(lessThan(coord,uvec2(WINDOW_X,WINDOW_Y))));
  compute(coord);
}

).";
