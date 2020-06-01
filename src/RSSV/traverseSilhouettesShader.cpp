#include <RSSV/traverseSilhouettesShader.h>

std::string const rssv::traverseSilhouettesShader = R".(

#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef NOF_TRIANGLES
#define NOF_TRIANGLES 0u
#endif//NOF_TRIANGLES

#ifndef SF_ALIGNMENT
#define SF_ALIGNMENT 128
#endif//SF_ALIGNMENT

#ifndef SF_INTERLEAVE
#define SF_INTERLEAVE 0
#endif//SF_INTERLEAVE

#ifndef NO_AABB
#define NO_AABB 0
#endif//NO_AABB

#pragma debug(on)

layout(local_size_x=WARP)in;

#if MERGED_BUFFERS == 1

layout(std430,binding=0)buffer Hierarchy{
  uint  nodePool[nodeBufferSizeInUints ];
  float aabbPool[aabbBufferSizeInFloats];
#if MEMORY_OPTIM == 1
  uint  aabbPointer[aabbPointerBufferSizeInUints];
#endif
};

#else
layout(std430,binding=0)buffer NodePool          {uint  nodePool         [];};
layout(std430,binding=1)buffer AABBPool          {float aabbPool         [];};
#if MEMORY_OPTIM == 1
layout(std430,binding=7)buffer AABBPointer {uint  aabbPointer [];};///TODO DEBUG???
#endif

#endif

layout(std430,binding=2)buffer JobCounter        {uint  jobCounter       [];};
layout(std430,binding=3)buffer EdgeBuffer        {float edgeBuffer       [];};
layout(std430,binding=4)buffer MultBuffer        {uint  multBuffer       [];};
layout(std430,binding=5)buffer SilhouetteCounter {uint  silhouetteCounter[];};
layout(std430,binding=6)buffer Bridges           { int  bridges          [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;
layout(r32i,binding=2)          uniform iimage2D      stencil     ;

uniform mat4 view;
uniform mat4 proj;
uniform vec4 lightPosition;

shared int  edgeMult;

uniform mat4 invTran;
uniform mat4 projView;

shared vec4 edgePlane;
shared vec4 aPlane   ;
shared vec4 bPlane   ;
shared vec4 abPlane  ;

shared vec4 edgeAClipSpace;
shared vec4 edgeBClipSpace;
shared vec4 lightClipSpace;

#line 52


#if STORE_EDGE_PLANES == 1
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif

void loadSilhouette(uint job){
  if(gl_LocalInvocationIndex == 0){
    uint res  = multBuffer[job];
    uint edge = res & 0x1fffffffu;
    int  mult = int(res) >> 29;

    vec3 edgeA;
    vec3 edgeB;
    edgeA[0] = edgeBuffer[edge+0*ALIGNED_NOF_EDGES];
    edgeA[1] = edgeBuffer[edge+1*ALIGNED_NOF_EDGES];
    edgeA[2] = edgeBuffer[edge+2*ALIGNED_NOF_EDGES];
    edgeB[0] = edgeBuffer[edge+3*ALIGNED_NOF_EDGES];
    edgeB[1] = edgeBuffer[edge+4*ALIGNED_NOF_EDGES];
    edgeB[2] = edgeBuffer[edge+5*ALIGNED_NOF_EDGES];

#if COMPUTE_PLANES_IN_CLIP_SPACE == 1
    edgeAClipSpace = projView*vec4(edgeA,1.f);
    edgeBClipSpace = projView*vec4(edgeB,1.f);
    lightClipSpace = projView*lightPosition  ;

    edgeAClipSpace /= abs(edgeAClipSpace.w);
    edgeBClipSpace /= abs(edgeBClipSpace.w);
    lightClipSpace /= abs(lightClipSpace.w);


    #if USE_SKALA == 1
      getEdgePlanesSkala(edgePlane,aPlane,bPlane,abPlane,edgeAClipSpace,edgeBClipSpace,lightClipSpace);
    #else
      getEdgePlanes(edgePlane,aPlane,bPlane,abPlane,edgeAClipSpace,edgeBClipSpace,lightClipSpace);
    #endif
#else
    vec3 n = normalize(cross(edgeB-edgeA,lightPosition.xyz-edgeA));
    edgePlane = invTran*vec4(n,-dot(n,edgeA));

    vec3 an = normalize(cross(n,edgeA-lightPosition.xyz));
    aPlane = invTran*vec4(an,-dot(an,edgeA));

    vec3 bn = normalize(cross(edgeB-lightPosition.xyz,n));
    bPlane = invTran*vec4(bn,-dot(bn,edgeB));

    vec3 abn = normalize(cross(edgeB-edgeA,n));
    abPlane = invTran*vec4(abn,-dot(abn,edgeA));

#if COMPUTE_BRIDGES == 1
    edgeAClipSpace = projView*vec4(edgeA,1.f);
    edgeBClipSpace = projView*vec4(edgeB,1.f);
    lightClipSpace = projView*lightPosition  ;
#endif

#endif

#if STORE_EDGE_PLANES == 1
    uint w = atomicAdd(debug[0],1);

    #if DUMP_POINTS_NOT_PLANES == 1
        for(int i=0;i<4;++i)
          debug[1+w*16+ 0+i] = floatBitsToUint(edgeAClipSpace[i]);
    
        for(int i=0;i<4;++i)
          debug[1+w*16+ 4+i] = floatBitsToUint(edgeBClipSpace[i]);
    
        for(int i=0;i<4;++i)
          debug[1+w*16+ 8+i] = floatBitsToUint(lightClipSpace[i]);
    
        for(int i=0;i<4;++i)
          debug[1+w*16+12+i] = floatBitsToUint(lightClipSpace[i]);
    #else
        for(int i=0;i<4;++i)
          debug[1+w*16+ 0+i] = floatBitsToUint(edgePlane[i]);
    
        for(int i=0;i<4;++i)
          debug[1+w*16+ 4+i] = floatBitsToUint(   aPlane[i]);
    
        for(int i=0;i<4;++i)
          debug[1+w*16+ 8+i] = floatBitsToUint(   bPlane[i]);
    
        for(int i=0;i<4;++i)
          debug[1+w*16+12+i] = floatBitsToUint(  abPlane[i]);
    #endif
#endif
    

    edgeMult = mult;
  }
  memoryBarrierShared();
}

int computeBridge(in vec3 bridgeStart,in vec3 bridgeEnd){
  // m n c 
  // - - 0
  // 0 - 1
  // + - 1
  // - 0 1
  // 0 0 0
  // + 0 0
  // - + 1
  // 0 + 0
  // + + 0
  //
  // m<0 && n>=0 || m>=0 && n<0
  // m<0 xor n<0

  int result = edgeMult;
  float ss = dot(edgePlane,vec4(bridgeStart,1));
  float es = dot(edgePlane,vec4(bridgeEnd  ,1));
  if((ss<0)==(es<0))return 0;
  result *= 1-2*int(ss<0.f);

  vec4 samplePlane    = getClipPlaneSkala(vec4(bridgeStart,1),vec4(bridgeEnd,1),lightClipSpace);
  ss = dot(samplePlane,edgeAClipSpace);
  es = dot(samplePlane,edgeBClipSpace);
  ss*=es;
  if(ss>0.f)return 0;
  result *= 1+int(ss<0.f);

  vec4 trianglePlane  = getClipPlaneSkala(vec4(bridgeStart,1),vec4(bridgeEnd,1),vec4(bridgeStart,1) + (edgeBClipSpace-edgeAClipSpace));
  trianglePlane *= sign(dot(trianglePlane,lightClipSpace));
  if(dot(trianglePlane,edgeAClipSpace)<=0)return 0;

  return result;

}

//void lastLevel(uint node){
//  uvec2 sampleCoord;
//  vec4 clipCoord;
//  bool inside;
//  vec4 plane;
//
//
//  sampleCoord = (demorton(node).xy<<uvec2(tileBitsX,tileBitsY)) + uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);
//
//  clipCoord.z = texelFetch(depthTexture,ivec2(sampleCoord)).x*2-1;
//  clipCoord.xy = -1+2*((vec2(sampleCoord) + vec2(0.5)) / vec2(WINDOW_X,WINDOW_Y));
//  clipCoord.w = 1.f;
//
//}

#if STORE_TRAVERSE_STAT == 1
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif

vec3 trivialRejectCorner3D(vec3 Normal){
  return vec3((ivec3(sign(Normal))+1)>>1);
}

#if STORE_TRAVERSE_STAT == 1
uint job = 0u;
#endif

#if WARP == 64
#line 10000

shared uint64_t intersection[nofLevels];

#if COMPUTE_BRIDGES == 1
#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
shared vec3     bridgeEnd   [nofLevels][WARP];
#endif
#endif

void traverse(){
  int level = 0;

  uint64_t currentIntersection;

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){

#if STORE_TRAVERSE_STAT == 1
      if(gl_LocalInvocationIndex==0){
        uint w = atomicAdd(debug[0],1);
        debug[1+w*4+0] = job;
        debug[1+w*4+1] = node;
        debug[1+w*4+2] = uint(level);
        debug[1+w*4+3] = 0xff;
      }
#endif

#if 1
      //lastLevel(node);
#endif
      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;

#if NO_AABB == 1
        vec3 minCorner;
        vec3 aabbSize;
        //*
        uvec3 xyz = (demorton(((node<<warpBits)+gl_LocalInvocationIndex)<<(warpBits*(nofLevels-1-level))));

        float startX = -1.f + xyz.x*levelTileSizeClipSpace[nofLevels-1].x;
        float startY = -1.f + xyz.y*levelTileSizeClipSpace[nofLevels-1].y;
        float endX   = min(startX + levelTileSizeClipSpace[level].x,1.f);
        float endY   = min(startY + levelTileSizeClipSpace[level].y,1.f);
        float startZ = Z_TO_DEPTH(CLUSTER_TO_Z(xyz.z                             ));
        float endZ   = Z_TO_DEPTH(CLUSTER_TO_Z(xyz.z+(1u<<levelTileBits[level].z)));

        minCorner[0] = startX;
        minCorner[1] = startY;
        minCorner[2] = startZ;

        aabbSize[0] = (endX-startX);
        aabbSize[1] = (endY-startY);
        aabbSize[2] = (endZ-startZ);

        // */
#else


        vec3 minCorner;
        vec3 aabbSize;
#if MEMORY_OPTIM == 1
        uint w = aabbPointer[nodeLevelOffset[level] + node*WARP + gl_LocalInvocationIndex + 1];
        minCorner[0] = aabbPool[w*6u + 0u]             ;
        minCorner[1] = aabbPool[w*6u + 2u]             ;
        minCorner[2] = aabbPool[w*6u + 4u]             ;
        aabbSize [0] = aabbPool[w*6u + 1u]-minCorner[0];
        aabbSize [1] = aabbPool[w*6u + 3u]-minCorner[1];
        aabbSize [2] = aabbPool[w*6u + 5u]-minCorner[2];
#else
        minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 0u]             ;
        minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 2u]             ;
        minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 4u]             ;
        aabbSize [0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 1u]-minCorner[0];
        aabbSize [1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 3u]-minCorner[1];
        aabbSize [2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 5u]-minCorner[2];
#endif

#endif

#if COMPUTE_BRIDGES == 1

#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
        bridgeEnd[level][gl_LocalInvocationIndex] = minCorner + aabbSize * 0.5f;

        vec3 bridgeStart;
        if(level == 0)
          bridgeStart = lightPosition.xyz;
        else
          bridgeStart = bridgeEnd[level-1][node&warpMask];
#endif

        if(level > 0){
#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
          int mult = computeBridge(bridgeStart,bridgeEnd[level][gl_LocalInvocationIndex]);
#else
          vec3 bridgeStart;
          bridgeStart[0]  = aabbPool[aabbLevelOffsetInFloats[level-1] + (node>>warpBits)*WARP*6u + gl_LocalInvocationIndex*6u + 0u];
          bridgeStart[1]  = aabbPool[aabbLevelOffsetInFloats[level-1] + (node>>warpBits)*WARP*6u + gl_LocalInvocationIndex*6u + 2u];
          bridgeStart[2]  = aabbPool[aabbLevelOffsetInFloats[level-1] + (node>>warpBits)*WARP*6u + gl_LocalInvocationIndex*6u + 4u];
          bridgeStart[0] += aabbPool[aabbLevelOffsetInFloats[level-1] + (node>>warpBits)*WARP*6u + gl_LocalInvocationIndex*6u + 1u];
          bridgeStart[1] += aabbPool[aabbLevelOffsetInFloats[level-1] + (node>>warpBits)*WARP*6u + gl_LocalInvocationIndex*6u + 3u];
          bridgeStart[2] += aabbPool[aabbLevelOffsetInFloats[level-1] + (node>>warpBits)*WARP*6u + gl_LocalInvocationIndex*6u + 5u];
          bridgeStart*=0.5;

          int mult = computeBridge(
              
              bridgeStart,
              
              minCorner + aabbSize/2.f);
#endif
          if(mult!=0)atomicAdd(bridges[nodeLevelOffsetInUints[level] + node*WARP + gl_LocalInvocationIndex],mult);
        }
#endif


        vec3 tr;
        bool planeTest;

#if 1
          status = TRIVIAL_REJECT;
          tr = trivialRejectCorner3D(edgePlane.xyz);
          if(dot(edgePlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f){
            if(dot(edgePlane,vec4(minCorner + (1.f-tr)*(aabbSize),1.f))<=0.f){
              tr = trivialRejectCorner3D(aPlane.xyz);
              if(dot(aPlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f){
                tr = trivialRejectCorner3D(bPlane.xyz);
                if(dot(bPlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f){
                  tr = trivialRejectCorner3D(abPlane.xyz);
                  if(dot(abPlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f)
                    status = INTERSECTS;
                }
              }
            }
          }
#endif

#if 0
        tr = trivialRejectCorner3D(edgePlane.xyz);
        planeTest =              dot(edgePlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f;
        planeTest = planeTest && dot(edgePlane,vec4(minCorner + (1.f-tr)*(aabbSize),1.f))<=0.f;
        tr = trivialRejectCorner3D(aPlane.xyz);
        planeTest = planeTest && dot(aPlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f;
        tr = trivialRejectCorner3D(bPlane.xyz);
        planeTest = planeTest && dot(bPlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f;
        tr = trivialRejectCorner3D(abPlane.xyz);
        planeTest = planeTest && dot(abPlane,vec4(minCorner + (    tr)*(aabbSize),1.f))>=0.f;

        if(planeTest)
          status = INTERSECTS;
        else
          status = TRIVIAL_REJECT;
#endif

      }

#if STORE_TRAVERSE_STAT == 1
        uint w = atomicAdd(debug[0],1);
        debug[1+w*4+0] = job;
        debug[1+w*4+1] = node*WARP + gl_LocalInvocationIndex;
        debug[1+w*4+2] = uint(level);
        debug[1+w*4+3] = status;
#endif

      currentIntersection = ballotARB(status == INTERSECTS    );
      if(gl_LocalInvocationIndex==0)
        intersection[level] = currentIntersection;

    }

    while(level >= 0 && currentIntersection == 0ul){
      node >>= warpBits;
      level--;
      if(level < 0)break;
      currentIntersection = intersection[level];
    }

    if(level < 0)break;
    if(level>=0){

      uint selectedBit = unpackUint2x32(currentIntersection)[0]!=0?findLSB(unpackUint2x32(currentIntersection)[0]):findLSB(unpackUint2x32(currentIntersection)[1])+32u;

      node <<= warpBits   ;
      node  += selectedBit;

      uint64_t mask = 1ul;
      mask <<= selectedBit;

      currentIntersection ^= mask;
      if(gl_LocalInvocationIndex==0)
        intersection[level] = currentIntersection;

      level++;
    }
  }
}

#endif

#line 36
void main(){
  #if STORE_TRAVERSE_STAT == 1
  #else
  uint job;
  #endif

  for(;;){
    if(gl_LocalInvocationIndex==0){
      job = atomicAdd(jobCounter[0],1);
    }

    job = readFirstInvocationARB(job);
    if(job >= silhouetteCounter[0])return;

    //job=3;
    loadSilhouette(job);

    traverse();
    //break;

  }
}
).";
