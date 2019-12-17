#include <Sintorn2/buildHierarchyShader.h>

std::string const sintorn2::reduceShader = R".(
shared float reductionArray[(TILE_X*TILE_Y)*3u];

#if WARP == 32
void reduce(){
  const uint halfWarp        = WARP / 2u;
  const uint halfWarpMask    = uint(halfWarp - 1u);

  float ab[2];
  uint w;

  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<3;++k){
  //    float mmin = +1e10;
  //    float mmax = -1e10;
  //    for(uint i=0;i<TILE_X*TILE_Y;++i){
  //      mmin = min(mmin,reductionArray[k*(TILE_X*TILE_Y)+i]);
  //      mmax = max(mmax,reductionArray[k*(TILE_X*TILE_Y)+i]);
  //    }
  //    reductionArray[k*2+0] = mmin;
  //    reductionArray[k*2+1] = mmax;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u)+ 0u];       
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u)+ 1u];       
  reductionArray[WARP*0u+gl_LocalInvocationIndex] = min(ab[0],ab[1]);
  reductionArray[WARP*1u+gl_LocalInvocationIndex] = max(ab[0],ab[1]);

  ab[0] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u)+ 0u];       
  ab[1] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u)+ 1u];       
  reductionArray[WARP*2u+gl_LocalInvocationIndex] = min(ab[0],ab[1]);
  reductionArray[WARP*3u+gl_LocalInvocationIndex] = max(ab[0],ab[1]);

  ab[0] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u)+ 0u];       
  ab[1] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u)+ 1u];       
  reductionArray[WARP*4u+gl_LocalInvocationIndex] = min(ab[0],ab[1]);
  reductionArray[WARP*5u+gl_LocalInvocationIndex] = max(ab[0],ab[1]);
memoryBarrierShared();//even if we have 32 threads WG == warp size of NVIDIA - barrier is necessary on 2080ti


  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<32;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*32+i]);
  //      else
  //        ext = max(ext,reductionArray[k*32+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;


  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>2u))!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>2u))!=0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*2u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>2u))!=0u)) > 0.f);
  reductionArray[WARP*2u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();

  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<16;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*16+i]);
  //      else
  //        ext = max(ext,reductionArray[k*16+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>3u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  ab[0] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*1u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>3u)) != 0u)) > 0.f);
  reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();

  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<8;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*8+i]);
  //      else
  //        ext = max(ext,reductionArray[k*8+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>4u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();


  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<4;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*4+i]);
  //      else
  //        ext = max(ext,reductionArray[k*4+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>5u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();
 
  //if(gl_LocalInvocationIndex == 0){
  //  for(uint k=0;k<6;++k){
  //    float ext = +1e10f * (-1+2*float((k%2)==0));
  //    for(uint i=0;i<2;++i){
  //      if((k%2) == 0)
  //        ext = min(ext,reductionArray[k*2+i]);
  //      else
  //        ext = max(ext,reductionArray[k*2+i]);
  //    }
  //    reductionArray[k] = ext;
  //  }
  //}
  //return;

  ab[0] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[(TILE_X*TILE_Y)*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&((TILE_X*TILE_Y)>>6u)) != 0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
memoryBarrierShared();
}
#endif

#if WARP == 64
void reduce(){
  const uint halfWarp        = WARP / 2u;
  const uint halfWarpMask    = uint(halfWarp - 1u);

  float ab[2];
  uint w;

  ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+ 0u     ];       
  ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+halfWarp];       
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&halfWarp)!=0u)) > 0.f);
  reductionArray[WARP*0u+gl_LocalInvocationIndex] = ab[w];                         

  ab[0] = reductionArray[WARP*1u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+ 0u     ];       
  ab[1] = reductionArray[WARP*1u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+halfWarp];       
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&halfWarp)!=0u)) > 0.f);
  reductionArray[WARP*1u+gl_LocalInvocationIndex] = ab[w];                         

  ab[0] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+ 0u     ];       
  ab[1] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)&halfWarpMask)+halfWarp];       
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&halfWarp)!=0u)) > 0.f);
  reductionArray[WARP*2u + gl_LocalInvocationIndex] = ab[w];                         



  ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
  ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
  w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>2u))!=0u)) > 0.f);
  reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];

  if(gl_LocalInvocationIndex < (WARP>>1u)){
    ab[0] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*2u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>2u))!=0u)) > 0.f);
    reductionArray[WARP*1u + gl_LocalInvocationIndex] = ab[w];
  }



  if((WARP>>3u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>3u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }

  if((WARP>>4u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>4u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }

  if((WARP>>5u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>5u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }
  
  if((WARP>>6u) > 0u){
    ab[0] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 0u];                 
    ab[1] = reductionArray[WARP*0u+(uint(gl_LocalInvocationIndex)<<1u) + 1u];                 
    w = uint((ab[1]-ab[0])*(-1.f+2.f*float((uint(gl_LocalInvocationIndex)&(WARP>>6u)) != 0u)) > 0.f);
    reductionArray[WARP*0u + gl_LocalInvocationIndex] = ab[w];
  }

}
#endif

).";

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

layout(std430,binding=0)buffer NodePool        {uint  nodePool        [];};
layout(std430,binding=1)buffer AABBPool        {float aabbPool        [];};
layout(std430,binding=3)buffer LevelNodeCounter{uint  levelNodeCounter[];};
layout(std430,binding=4)buffer ActiveNodes     {uint  activeNodes     [];};

layout(binding=1)uniform sampler2DRect depthTexture;

uint getMorton(uvec2 coord,float depth){
  const uint tileBitsX     = uint(ceil(log2(float(TILE_X))));
  const uint tileBitsY     = uint(ceil(log2(float(TILE_Y))));

  float z = depthToZ(depth);
  uint  zQ = quantizeZ(z);
  uvec3 clusterCoord = uvec3(uvec2(coord) >> uvec2(tileBitsX,tileBitsY), zQ);
  return morton(clusterCoord);
}



#if WARP == 64
uint activeThread = 0;
#else
uint activeThread[2] = {0,0};
#endif


#if WARP == 64
void compute(uvec2 coord){
#else
void compute(uvec2 coord,uvec2 coord2){
#endif




#if WARP==64
  float depth = texelFetch(depthTexture,ivec2(coord)).x*2-1;
  uint morton = getMorton(coord,depth);
  if(depth >= 1.f)activeThread = 0;
#else
  float depth [2];
  uint  morton[2];
  depth [0] = texelFetch(depthTexture,ivec2(coord )).x*2-1;
  depth [1] = texelFetch(depthTexture,ivec2(coord2)).x*2-1;
  morton[0] = getMorton(coord ,depth[0]);
  morton[1] = getMorton(coord2,depth[1]);
#endif

#line 322
  //if(uintsPerWarp == 1){
  #if WARP == 32
    uint counter = 0;
    uint notDone[2];
    notDone[0] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread[0] != 0)),0);
    notDone[1] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(activeThread[1] != 0)),0);
    while(notDone[0] != 0 || notDone[1] != 0){

      if(counter >= (TILE_X*TILE_Y))break;
      counter ++;

      uint selectedBit     = notDone[0]!=0?findLSB(notDone[0]):findLSB(notDone[1])+32u;
      uint referenceMorton = readInvocationARB(morton[uint(selectedBit>31u)],selectedBit&uint(0x1fu));

      if(gl_LocalInvocationIndex == 0){
        if(nofLevels>0){
          uint bit  = (referenceMorton >> (warpBits*0u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*1u));
          atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-1u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>1){
          uint bit  = (referenceMorton >> (warpBits*1u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*2u));
          uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-2u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
          if(mm == 0){
            mm = atomicAdd(levelNodeCounter[clamp(nofLevels-2u,0u,5u)*4u],1);
            activeNodes[nodeLevelOffset[clamp(nofLevels-2u,0u,5u)]+mm] = node*uintsPerWarp;
          }
        }
        //if(nofLevels>2){
        //  uint bit  = (referenceMorton >> (warpBits*2u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*3u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-3u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
        //if(nofLevels>3){
        //  uint bit  = (referenceMorton >> (warpBits*3u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*4u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-4u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
        //if(nofLevels>4){
        //  uint bit  = (referenceMorton >> (warpBits*4u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*5u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-5u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
        //if(nofLevels>5){
        //  uint bit  = (referenceMorton >> (warpBits*5u)) & warpMask;
        //  uint node = (referenceMorton >> (warpBits*6u));
        //  atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-6u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        //}
      }

      uint sameCluster[2];
      sameCluster[0] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton[0])),0);
      sameCluster[1] = GET_UINT_FROM_UINT_ARRAY(BALLOT_RESULT_TO_UINTS(BALLOT(referenceMorton == morton[1])),0);

      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+0   ] = -1.f + 2.f/float(WINDOW_X)*(coord.x+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+0   ] = -1.f + 2.f/float(WINDOW_Y)*(coord.y+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+0   ] = depth[0];
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+WARP] = -1.f + 2.f/float(WINDOW_X)*(coord2.x+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+WARP] = -1.f + 2.f/float(WINDOW_Y)*(coord2.y+0.5f);
      reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+WARP] = depth[1];

      memoryBarrierShared();

      if(referenceMorton != morton[0] || activeThread[0] == 0){
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+0] = reductionArray[selectedBit+(TILE_X*TILE_Y)*0u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+0] = reductionArray[selectedBit+(TILE_X*TILE_Y)*1u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+0] = reductionArray[selectedBit+(TILE_X*TILE_Y)*2u];
      }

      if(referenceMorton != morton[1] || activeThread[1] == 0){
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*0u+WARP] = reductionArray[selectedBit+(TILE_X*TILE_Y)*0u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*1u+WARP] = reductionArray[selectedBit+(TILE_X*TILE_Y)*1u];
        reductionArray[gl_LocalInvocationIndex+(TILE_X*TILE_Y)*2u+WARP] = reductionArray[selectedBit+(TILE_X*TILE_Y)*2u];
      }
      memoryBarrierShared();

      reduce();

      if(gl_LocalInvocationIndex < floatsPerAABB){
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+gl_LocalInvocationIndex] = reductionArray[gl_LocalInvocationIndex];
      }

      notDone[0] ^= sameCluster[0];
      notDone[1] ^= sameCluster[1];
    }
  #endif
  //}
 

#line 166
  #if WARP == 64
    uint counter = 0;
    uint64_t notDone = ballotARB(activeThread != 0);
    while(notDone != 0){

      if(counter >= WARP)break;

      uint selectedBit     = unpackUint2x32(notDone)[0]!=0?findLSB(unpackUint2x32(notDone)[0]):findLSB(unpackUint2x32(notDone)[1])+32u;
      uint referenceMorton = readInvocationARB(morton,selectedBit);

      if(gl_LocalInvocationIndex == 0){

        if(nofLevels>0){
          uint bit  = (referenceMorton >> (warpBits*0u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*1u));
          uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-1u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
        }
        if(nofLevels>1){
          uint bit  = (referenceMorton >> (warpBits*1u)) & warpMask;
          uint node = (referenceMorton >> (warpBits*2u));
          uint mm = atomicOr(nodePool[nodeLevelOffsetInUints[clamp(nofLevels-2u,0u,5u)]+node*uintsPerWarp+uint(bit>31u)],1u<<(bit&0x1fu));
          if(mm == 0){
            mm = atomicAdd(levelNodeCounter[clamp(nofLevels-2u,0u,5u)*4u],1);
            activeNodes[nodeLevelOffset[clamp(nofLevels-2u,0u,5u)]+mm] = node*uintsPerWarp+uint(bit>31u);
          }
        }

      }
      
      uint64_t sameCluster = ballotARB(referenceMorton == morton && activeThread != 0);

      reductionArray[gl_LocalInvocationIndex+WARP*0u] = -1.f + 2.f/float(WINDOW_X)*(coord.x+0.5f);
      reductionArray[gl_LocalInvocationIndex+WARP*1u] = -1.f + 2.f/float(WINDOW_Y)*(coord.y+0.5f);
      reductionArray[gl_LocalInvocationIndex+WARP*2u] = depth;

      if(referenceMorton != morton || activeThread == 0){
        reductionArray[gl_LocalInvocationIndex+WARP*0u] = reductionArray[selectedBit+WARP*0u];
        reductionArray[gl_LocalInvocationIndex+WARP*1u] = reductionArray[selectedBit+WARP*1u];
        reductionArray[gl_LocalInvocationIndex+WARP*2u] = reductionArray[selectedBit+WARP*2u];
      }

      reduce();

      if(gl_LocalInvocationIndex < floatsPerAABB){
        uint node = (referenceMorton >> (warpBits*0u));
        aabbPool[aabbLevelOffsetInFloats[clamp(nofLevels-1u,0u,5u)]+node*floatsPerAABB+gl_LocalInvocationIndex] = reductionArray[gl_LocalInvocationIndex];
      }

      notDone ^= sameCluster;
      counter++;
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
  activeThread = uint(all(lessThan(coord,uvec2(WINDOW_X,WINDOW_Y))));
  compute(coord);
#else
  uvec2 coord  = wgCoord + loCoord;
  uvec2 coord2 = wgCoord + loCoord + uvec2(0,4u);
  activeThread[0] = uint(all(lessThan(coord ,uvec2(WINDOW_X,WINDOW_Y))));
  activeThread[1] = uint(all(lessThan(coord2,uvec2(WINDOW_X,WINDOW_Y))));
  compute(coord,coord2);
#endif
}

).";
