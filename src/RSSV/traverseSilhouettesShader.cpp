#include <RSSV/traverseSilhouettesShader.h>

std::string const rssv::traverseSilhouettesShader = 
#if 0
R".(

#ifndef WARP
#define WARP 64
#endif//WARP

//#pragma debug(on)

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer Hierarchy{
  uint  nodePool[nodeBufferSizeInUints ];
  float aabbPool[aabbBufferSizeInFloats];
  #if MEMORY_OPTIM == 1
    uint  aabbPointer[aabbPointerBufferSizeInUints];
  #endif
  #if USE_BRIDGE_POOL == 1
    float bridgePool[bridgePoolSizeInFloats];
  #endif
};

layout(std430,binding=2)buffer JobCounters       {
  uint silhouetteJobCounter;
  uint triangleJobCounter  ;
};

layout(std430,binding=3)readonly buffer EdgePlanes{float edgePlanes       [];};
layout(std430,binding=4)readonly buffer MultBuffer{
  uint nofSilhouettes  ;
  uint multBuffer    [];
};

#ifndef NOF_TRIANGLES
#define NOF_TRIANGLES 0u
#endif//NOF_TRIANGLES

#ifndef SF_ALIGNMENT
#define SF_ALIGNMENT 128
#endif//SF_ALIGNMENT

#ifndef SF_INTERLEAVE
#define SF_INTERLEAVE 0
#endif//SF_INTERLEAVE

#ifndef MORE_PLANES
#define MORE_PLANES 0
#endif//MORE_PLANES

layout(std430,binding=5)readonly buffer ShadowFrusta{float shadowFrusta[];};

layout(std430,binding=6)buffer Bridges           { int  bridges          [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;
layout(r32i,binding=2)          uniform iimage2D      stencil     ;

uniform vec4 lightPosition;
uniform vec4 clipLightPosition;

uniform mat4 invTran;
uniform mat4 projView;

const uint planesPerSF = 4u + MORE_PLANES*3u;
const uint floatsPerPlane = 4u;
const uint floatsPerSF = planesPerSF * floatsPerPlane;

const uint alignedNofSF = (uint(NOF_TRIANGLES / SF_ALIGNMENT) + uint((NOF_TRIANGLES % SF_ALIGNMENT) != 0u)) * SF_ALIGNMENT;

shared float shadowFrustaPlanes[floatsPerSF];


void loadShadowFrustum(uint job){
  if(gl_LocalInvocationIndex < floatsPerSF){
#if SF_INTERLEAVE == 1
    shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[alignedNofSF*gl_LocalInvocationIndex + job];
#else
    shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[job*floatsPerSF+gl_LocalInvocationIndex];
#endif
  }
#if WARP == 32
  memoryBarrierShared();
#endif
}

vec3 trivialRejectCorner3D(vec3 Normal){
  return vec3((ivec3(sign(Normal))+1)/2);
}


uint job = 0u;

uint trivialRejectAccept(vec3 minCorner,vec3 size){
  uint status = TRIVIAL_ACCEPT;
  vec4 plane;
  vec3 tr;
  if(minCorner.x != 1337)return TRIVIAL_REJECT;

  plane = vec4(shadowFrustaPlanes[0],shadowFrustaPlanes[1],shadowFrustaPlanes[2],shadowFrustaPlanes[3]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[4],shadowFrustaPlanes[5],shadowFrustaPlanes[6],shadowFrustaPlanes[7]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[8],shadowFrustaPlanes[9],shadowFrustaPlanes[10],shadowFrustaPlanes[11]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[12],shadowFrustaPlanes[13],shadowFrustaPlanes[14],shadowFrustaPlanes[15]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1.f),plane)>0.f);

  return status;
}

shared uint64_t intersection[nofLevels];

void traverse(){
  int level = 0;

  uint64_t currentIntersection;

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){

      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;

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

        status = trivialRejectAccept(minCorner,aabbSize);

      }


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








void main(){
  if(texelFetch(depthTexture,ivec2(0,0)).x != 1337)return;
  if(lightPosition.x == 1337)return;
  if(projView[0][0] == 1337)return;
  if(invTran[0][0] == 1337)return;
  if(clipLightPosition.x == 1337)return;
  if(lightPosition.y != 1337)return;

  for(;;){
    if(gl_LocalInvocationIndex==0){
      job = atomicAdd(triangleJobCounter,1);
    }

    job = readFirstInvocationARB(job);
    if(job >= NOF_TRIANGLES)return;

    loadShadowFrustum(job);

    traverse();

  }
}


).";
#endif








/////////////////////////////
// WORK IN PROGRESS TRIANGLES
/////////////////////////////
#if 0
R".(

#ifndef WARP
#define WARP 64
#endif//WARP

//#pragma debug(on)

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer Hierarchy{
  uint  nodePool[nodeBufferSizeInUints ];
  float aabbPool[aabbBufferSizeInFloats];
  #if MEMORY_OPTIM == 1
    uint  aabbPointer[aabbPointerBufferSizeInUints];
  #endif
  #if USE_BRIDGE_POOL == 1
    float bridgePool[bridgePoolSizeInFloats];
  #endif
};

layout(std430,binding=2)buffer JobCounters       {
  uint silhouetteJobCounter;
  uint triangleJobCounter  ;
};

layout(std430,binding=3)readonly buffer EdgePlanes{float edgePlanes       [];};
layout(std430,binding=4)readonly buffer MultBuffer{
  uint nofSilhouettes  ;
  uint multBuffer    [];
};

#ifndef NOF_TRIANGLES
#define NOF_TRIANGLES 0u
#endif//NOF_TRIANGLES

#ifndef SF_ALIGNMENT
#define SF_ALIGNMENT 128
#endif//SF_ALIGNMENT

#ifndef SF_INTERLEAVE
#define SF_INTERLEAVE 0
#endif//SF_INTERLEAVE

#ifndef MORE_PLANES
#define MORE_PLANES 0
#endif//MORE_PLANES

const uint planesPerSF = 4u + MORE_PLANES*3u;
const uint floatsPerPlane = 4u;
const uint floatsPerSF = planesPerSF * floatsPerPlane;
layout(std430,binding=5)readonly buffer ShadowFrusta{float shadowFrusta[];};

layout(std430,binding=6)buffer Bridges           { int  bridges          [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;
layout(r32i,binding=2)          uniform iimage2D      stencil     ;

uniform vec4 lightPosition;
uniform vec4 clipLightPosition;

uniform mat4 invTran;
uniform mat4 projView;

#if 0
shared int  edgeMult;

shared vec4 sharedVec4[7];

#define edgePlane      sharedVec4[0]
#define aPlane         sharedVec4[1]
#define bPlane         sharedVec4[2]
#define abPlane        sharedVec4[3]
#define edgeAClipSpace sharedVec4[4]
#define edgeBClipSpace sharedVec4[5]
#define lightClipSpace sharedVec4[6]

#define tri_trianglePlane sharedVec4[0]
#define tri_abPlane       sharedVec4[1]
#define tri_bcPlane       sharedVec4[2]
#define tri_caPlane       sharedVec4[3]
#endif

#if (STORE_EDGE_PLANES == 1) || (STORE_TRAVERSE_STAT == 1)
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif

#if 0
void loadSilhouette(uint job){
  if(gl_LocalInvocationIndex == 0){
    uint res  = multBuffer[job];
    uint edge = res & 0x1fffffffu;
    int  mult = int(res) >> 29;

    vec3 edgeA;
    vec3 edgeB;
    loadEdge(edgeA,edgeB,edge);

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
    lightClipSpace = clipLightPosition;
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

int computeBridgeSilhouetteMultiplicity(in vec4 bridgeStart,in vec4 bridgeEnd){
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
  float ss = dot(edgePlane,bridgeStart);
  float es = dot(edgePlane,bridgeEnd  );
  if((ss<0)==(es<0))return 0;
  result *= 1-2*int(ss<0.f);

  vec4 samplePlane    = getClipPlaneSkala(bridgeStart,bridgeEnd,lightClipSpace);
  ss = dot(samplePlane,edgeAClipSpace);
  es = dot(samplePlane,edgeBClipSpace);
  ss*=es;
  if(ss>0.f)return 0;
  result *= 1+int(ss<0.f);

  vec4 trianglePlane  = getClipPlaneSkala(bridgeStart,bridgeEnd,bridgeStart + (edgeBClipSpace-edgeAClipSpace));
  trianglePlane *= sign(dot(trianglePlane,lightClipSpace));
  if(dot(trianglePlane,edgeAClipSpace)<=0)return 0;

  return result;
}

void lastLevelSilhouette(uint node){
#if COMPUTE_LAST_LEVEL_SILHOUETTES == 1
  uvec2 sampleCoord = (demorton(node).xy<<uvec2(tileBitsX,tileBitsY)) + uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);

  vec4 bridgeEnd;
  bridgeEnd.z = texelFetch(depthTexture,ivec2(sampleCoord)).x*2-1;
  bridgeEnd.xy = -1+2*((vec2(sampleCoord) + vec2(0.5)) / vec2(WINDOW_X,WINDOW_Y));
  bridgeEnd.w = 1.f;

  vec4 bridgeStart = vec4(getAABBCenter(nofLevels-1,node),1.f);

  int mult = computeBridgeSilhouetteMultiplicity(bridgeStart,bridgeEnd);

  if(mult!=0)imageAtomicAdd(stencil,ivec2(sampleCoord),mult);
#endif
}

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

#if 0
#if COMPUTE_BRIDGES == 1
#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
shared vec3     localBridgeEnd   [nofLevels][WARP];
#endif
#endif

void debug_storeSilhouetteTraverseStatLastLevel(){
#if STORE_TRAVERSE_STAT == 1
  if(gl_LocalInvocationIndex==0){
    uint w = atomicAdd(debug[0],1);
    debug[1+w*4+0] = job;
    debug[1+w*4+1] = node;
    debug[1+w*4+2] = uint(level);
    debug[1+w*4+3] = 0xff;
  }
#endif
}

void debug_storeSilhouetteTraverseStat(){
#if STORE_TRAVERSE_STAT == 1
  uint w = atomicAdd(debug[0],1);
  debug[1+w*4+0] = job;
  debug[1+w*4+1] = node*WARP + gl_LocalInvocationIndex;
  debug[1+w*4+2] = uint(level);
  debug[1+w*4+3] = status;
#endif
}

void computeBridgeSilhouetteIntersection(in vec3 minCorner,in vec3 aabbSize,int level,uint node){
#if COMPUTE_BRIDGES == 1
  vec4 bridgeStart;
  vec4 bridgeEnd  ;
  int  mult       ;


  bridgeEnd = vec4(minCorner + aabbSize*.5f,1.f);

#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
  localBridgeEnd[level][gl_LocalInvocationIndex] = vec3(bridgeEnd);
#endif

  if(level == 0){
    bridgeStart = clipLightPosition;
  }else{
#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
    bridgeStart = vec4(localBridgeEnd[level-1][node&warpMask],1);
#else
    bridgeStart = vec4(getAABBCenter(level-1,node),1);
#endif
  }

  mult = computeBridgeSilhouetteMultiplicity(bridgeStart,bridgeEnd);
  if(mult!=0)atomicAdd(bridges[nodeLevelOffset[level] + node*WARP + gl_LocalInvocationIndex],mult);
#endif
}

void computeAABBSilhouetteIntersection(out uint status,in vec3 minCorner,in vec3 aabbSize){
  status = TRIVIAL_REJECT;

  vec3 tr;

  tr = trivialRejectCorner3D(edgePlane.xyz);
  if(dot(edgePlane,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  tr = 1.f-tr;
  if(dot(edgePlane,vec4(minCorner + (tr)*(aabbSize),1.f))>0.f)return;

  tr = trivialRejectCorner3D(aPlane.xyz);
  if(dot(aPlane   ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  tr = trivialRejectCorner3D(bPlane.xyz);
  if(dot(bPlane   ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  tr = trivialRejectCorner3D(abPlane.xyz);
  if(dot(abPlane  ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  status = INTERSECTS;
}

void traverseSilhouette(){
  int level = 0;

  uint64_t currentIntersection;

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){

      debug_storeSilhouetteTraverseStatLastLevel();

      lastLevelSilhouette(node);

      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;

        vec3 minCorner;
        vec3 aabbSize;
        getAABB(minCorner,aabbSize,level,(node<<warpBits)+gl_LocalInvocationIndex);
        aabbSize -= minCorner;

        computeBridgeSilhouetteIntersection(minCorner,aabbSize,level,node);

        computeAABBSilhouetteIntersection(status,minCorner,aabbSize);

      }

      debug_storeSilhouetteTraverseStat();

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

#endif

const uint alignedNofSF        = (uint(NOF_TRIANGLES /       SF_ALIGNMENT) + uint((NOF_TRIANGLES %       SF_ALIGNMENT) != 0u)) *       SF_ALIGNMENT;

shared float shadowFrustaPlanes[floatsPerSF];

#if 1
void loadTriangle(uint job){
  if(gl_LocalInvocationIndex < floatsPerSF){
#if SF_INTERLEAVE == 1
    shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[alignedNofSF*gl_LocalInvocationIndex + job];
#else
    shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[job*floatsPerSF+gl_LocalInvocationIndex];
#endif
  }
#if WARP == 32
  memoryBarrierShared();
#endif
//  if(gl_LocalInvocationIndex == 0){
//#if SF_INTERLEAVE == 1
//    tri_trianglePlane[0] = shadowFrusta[alignedNofSF*0  + job];
//    tri_trianglePlane[1] = shadowFrusta[alignedNofSF*1  + job];
//    tri_trianglePlane[2] = shadowFrusta[alignedNofSF*2  + job];
//    tri_trianglePlane[3] = shadowFrusta[alignedNofSF*3  + job];
//    tri_abPlane      [0] = shadowFrusta[alignedNofSF*4  + job];
//    tri_abPlane      [1] = shadowFrusta[alignedNofSF*5  + job];
//    tri_abPlane      [2] = shadowFrusta[alignedNofSF*6  + job];
//    tri_abPlane      [3] = shadowFrusta[alignedNofSF*7  + job];
//    tri_bcPlane      [0] = shadowFrusta[alignedNofSF*8  + job];
//    tri_bcPlane      [1] = shadowFrusta[alignedNofSF*9  + job];
//    tri_bcPlane      [2] = shadowFrusta[alignedNofSF*10 + job];
//    tri_bcPlane      [3] = shadowFrusta[alignedNofSF*11 + job];
//    tri_caPlane      [0] = shadowFrusta[alignedNofSF*12 + job];
//    tri_caPlane      [1] = shadowFrusta[alignedNofSF*13 + job];
//    tri_caPlane      [2] = shadowFrusta[alignedNofSF*14 + job];
//    tri_caPlane      [3] = shadowFrusta[alignedNofSF*15 + job];
//#else
//    tri_trianglePlane[0] = shadowFrusta[job*floatsPerSF+0 ];
//    tri_trianglePlane[1] = shadowFrusta[job*floatsPerSF+1 ];
//    tri_trianglePlane[2] = shadowFrusta[job*floatsPerSF+2 ];
//    tri_trianglePlane[3] = shadowFrusta[job*floatsPerSF+3 ];
//    tri_abPlane      [0] = shadowFrusta[job*floatsPerSF+4 ];
//    tri_abPlane      [1] = shadowFrusta[job*floatsPerSF+5 ];
//    tri_abPlane      [2] = shadowFrusta[job*floatsPerSF+6 ];
//    tri_abPlane      [3] = shadowFrusta[job*floatsPerSF+7 ];
//    tri_bcPlane      [0] = shadowFrusta[job*floatsPerSF+8 ];
//    tri_bcPlane      [1] = shadowFrusta[job*floatsPerSF+9 ];
//    tri_bcPlane      [2] = shadowFrusta[job*floatsPerSF+10];
//    tri_bcPlane      [3] = shadowFrusta[job*floatsPerSF+11];
//    tri_caPlane      [0] = shadowFrusta[job*floatsPerSF+12];
//    tri_caPlane      [1] = shadowFrusta[job*floatsPerSF+13];
//    tri_caPlane      [2] = shadowFrusta[job*floatsPerSF+14];
//    tri_caPlane      [3] = shadowFrusta[job*floatsPerSF+15];
//#endif
//  }
//  memoryBarrierShared();
////  if(gl_LocalInvocationIndex == 0){
////    vec3 A;
////    vec3 B;
////    vec3 C;
////    A[0] = triangles[job*9+0*3+0];
////    A[1] = triangles[job*9+0*3+1];
////    A[2] = triangles[job*9+0*3+2];
////    B[0] = triangles[job*9+1*3+0];
////    B[1] = triangles[job*9+1*3+1];
////    B[2] = triangles[job*9+1*3+2];
////    C[0] = triangles[job*9+2*3+0];
////    C[1] = triangles[job*9+2*3+1];
////    C[2] = triangles[job*9+2*3+2];
////
////    //if(clipLightPosition.x == 1337)return;
////
////    vec3 n = normalize(cross(B-A,C-A));
////    tri_trianglePlane = invTran*vec4(n,-dot(n,A));
////    vec3 n2;
////    n2 = normalize(cross(n,B-A));
////    tri_abPlane = invTran*vec4(n2,-dot(n2,A));
////    n2 = normalize(cross(n,C-B));
////    tri_bcPlane = invTran*vec4(n2,-dot(n2,B));
////    n2 = normalize(cross(n,A-C));
////    tri_caPlane = invTran*vec4(n2,-dot(n2,C));
////    if(dot(tri_trianglePlane,clipLightPosition)<0){
////      tri_trianglePlane *= -1;
////      n *= -1;
////    }
////    tri_abPlane *= -1;
////    tri_bcPlane *= -1;
////    tri_caPlane *= -1;
////
////  }
////  memoryBarrierShared();
}

uint trivialRejectAccept(vec3 minCorner,vec3 size){
  uint status = TRIVIAL_ACCEPT;
  vec4 plane;
  vec3 tr;
  //if(minCorner.x != 1337)return TRIVIAL_REJECT;

  plane = vec4(shadowFrustaPlanes[0],shadowFrustaPlanes[1],shadowFrustaPlanes[2],shadowFrustaPlanes[3]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[4],shadowFrustaPlanes[5],shadowFrustaPlanes[6],shadowFrustaPlanes[7]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[8],shadowFrustaPlanes[9],shadowFrustaPlanes[10],shadowFrustaPlanes[11]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[12],shadowFrustaPlanes[13],shadowFrustaPlanes[14],shadowFrustaPlanes[15]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1.f),plane)>0.f);

  return status;
}

//void computeAABBTriangleIntersection(out uint status,in vec3 minCorner,in vec3 aabbSize){
//  status = TRIVIAL_REJECT;
//
//  vec3 tr;
//
//  //if(tri_trianglePlane.x != 1337)return;
//  //if(tri_abPlane.x != 1337)return;
//  //if(tri_bcPlane.x != 1337)return;
//  //if(tri_caPlane.x != 1337)return;
//
//  /*
//  tr = trivialRejectCorner3D(tri_trianglePlane.xyz);
//  if(dot(tri_trianglePlane,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;
//
//  tr = 1.f-tr;
//  if(dot(tri_trianglePlane,vec4(minCorner + (tr)*(aabbSize),1.f))>0.f)return;
//
//  tr = trivialRejectCorner3D(tri_abPlane.xyz);
//  if(dot(tri_abPlane      ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;
//
//  tr = trivialRejectCorner3D(tri_bcPlane.xyz);
//  if(dot(tri_bcPlane      ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;
//
//  tr = trivialRejectCorner3D(tri_caPlane.xyz);
//  if(dot(tri_caPlane      ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;
//  */
//
//  //if(tri_caPlane.w == 1337)return;
//  vec4 plane;
//  plane = tri_abPlane;
//  tr    = trivialRejectCorner3D(plane.xyz);
//  if(dot(plane,vec4(minCorner + tr*aabbSize,1.f))<0.f)return;
//
//  plane = tri_bcPlane;
//  tr    = trivialRejectCorner3D(plane.xyz);
//  if(dot(plane,vec4(minCorner + tr*aabbSize,1.f))<0.f)return;
//
//  plane = tri_caPlane;
//  tr    = trivialRejectCorner3D(plane.xyz);
//  if(dot(plane,vec4(minCorner + tr*aabbSize,1.f))<0.f)return;
//
//  plane = tri_trianglePlane;
//  tr    = trivialRejectCorner3D(plane.xyz);
//  if(dot(plane,vec4(minCorner + tr*aabbSize,1.f))<0.f)return;
//
//  tr    = 1.f - tr;
//  if(dot(plane,vec4(minCorner + tr*aabbSize,1.f))>0.f)return;
//
//  status = INTERSECTS;
//}

void traverseTriangle(){
  int level = 0;

  uint64_t currentIntersection;

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){

      //debug_storeSilhouetteTraverseStatLastLevel();

      //lastLevelTriangles(node);

      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;

        vec3 minCorner;
        vec3 aabbSize;
        getAABB(minCorner,aabbSize,level,(node<<warpBits)+gl_LocalInvocationIndex);
        aabbSize -= minCorner;

        //computeBridgeSilhouetteIntersection(minCorner,aabbSize,level,node);

        //computeAABBTriangleIntersection(status,minCorner,aabbSize);
        status = trivialRejectAccept(minCorner,aabbSize);
        if(status != INTERSECTS)status = TRIVIAL_REJECT;

        //if(gl_LocalInvocationIndex > 31)status = TRIVIAL_REJECT;
        //computeAABBSilhouetteIntersection(status,minCorner,aabbSize);
        //status == 0u;//TODO REMOVE

      }

      //debug_storeSilhouetteTraverseStat();

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

//uniform int selectedEdge = -1;

void main(){
  #if STORE_TRAVERSE_STAT != 1
  uint job;
  #endif

#if 0
  //silhouette loop
  for(;;){
    if(gl_LocalInvocationIndex==0)
      job = atomicAdd(silhouetteJobCounter,1);

    job = readFirstInvocationARB(job);
    if(job >= nofSilhouettes)break;

    //if(selectedEdge>=0 && job != uint(selectedEdge))continue;

    loadSilhouette(job);
    traverseSilhouette();

  }
#endif
  if(lightPosition.x == 1337)return;
  if(projView[0][0] == 1337)return;
  if(invTran[0][0] == 1337)return;
  if(clipLightPosition.x == 1337)return;

#if 1
  //triangle loop
  for(;;){
  
    if(gl_LocalInvocationIndex==0)
      job = atomicAdd(triangleJobCounter,1);

    job = readFirstInvocationARB(job);
    if(job >= NOF_TRIANGLES)break;

    //if(selectedEdge>=0 && job != uint(selectedEdge))continue;

    loadTriangle(job);
    traverseTriangle();
  
  }
#endif
}



).";

#endif 



















/////////////////////
/////////////////////
//WORKING silhouettes
/////////////////////
/////////////////////
#if 1
R".(

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer Hierarchy{
  uint  nodePool[nodeBufferSizeInUints ];
  float aabbPool[aabbBufferSizeInFloats];
  #if MEMORY_OPTIM == 1
    uint  aabbPointer[aabbPointerBufferSizeInUints];
  #endif
  #if USE_BRIDGE_POOL == 1
    float bridgePool[bridgePoolSizeInFloats];
  #endif
};

layout(std430,binding=2)buffer JobCounters       {
  uint silhouetteJobCounter;
  uint triangleJobCounter  ;
};

layout(std430,binding=3)readonly buffer EdgePlanes{float edgePlanes       [];};
layout(std430,binding=4)readonly buffer MultBuffer{
  uint nofSilhouettes  ;
  uint multBuffer    [];
};

layout(std430,binding=5)readonly buffer ShadowFrusta{float shadowFrusta[];};

layout(std430,binding=6)buffer Bridges           { int  bridges          [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;
layout(r32i,binding=2)          uniform iimage2D      stencil     ;

uniform vec4 lightPosition;
uniform vec4 clipLightPosition;

uniform mat4 invTran;
uniform mat4 projView;

shared int  edgeMult;

shared vec4 sharedVec4[7];

#define edgePlane      sharedVec4[0]
#define aPlane         sharedVec4[1]
#define bPlane         sharedVec4[2]
#define abPlane        sharedVec4[3]
#define edgeAClipSpace sharedVec4[4]
#define edgeBClipSpace sharedVec4[5]
#define lightClipSpace sharedVec4[6]

#if (STORE_EDGE_PLANES == 1) || (STORE_TRAVERSE_STAT == 1)
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif

void loadSilhouette(uint job){
  if(gl_LocalInvocationIndex == 0){
    uint res  = multBuffer[job];
    uint edge = res & 0x1fffffffu;
    int  mult = int(res) >> 29;

    vec3 edgeA;
    vec3 edgeB;
    loadEdge(edgeA,edgeB,edge);

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
    lightClipSpace = clipLightPosition;
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

int computeBridgeSilhouetteMultiplicity(in vec4 bridgeStart,in vec4 bridgeEnd){
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
  float ss = dot(edgePlane,bridgeStart);
  float es = dot(edgePlane,bridgeEnd  );
  if((ss<0)==(es<0))return 0;
  result *= 1-2*int(ss<0.f);

  vec4 samplePlane    = getClipPlaneSkala(bridgeStart,bridgeEnd,lightClipSpace);
  ss = dot(samplePlane,edgeAClipSpace);
  es = dot(samplePlane,edgeBClipSpace);
  ss*=es;
  if(ss>0.f)return 0;
  result *= 1+int(ss<0.f);

  vec4 trianglePlane  = getClipPlaneSkala(bridgeStart,bridgeEnd,bridgeStart + (edgeBClipSpace-edgeAClipSpace));
  trianglePlane *= sign(dot(trianglePlane,lightClipSpace));
  if(dot(trianglePlane,edgeAClipSpace)<=0)return 0;

  return result;
}

void lastLevelSilhouette(uint node){
#if COMPUTE_LAST_LEVEL_SILHOUETTES == 1
  uvec2 sampleCoord = (demorton(node).xy<<uvec2(tileBitsX,tileBitsY)) + uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);

  vec4 bridgeEnd;
  bridgeEnd.z = texelFetch(depthTexture,ivec2(sampleCoord)).x*2-1;
  bridgeEnd.xy = -1+2*((vec2(sampleCoord) + vec2(0.5)) / vec2(WINDOW_X,WINDOW_Y));
  bridgeEnd.w = 1.f;

  vec4 bridgeStart = vec4(getAABBCenter(nofLevels-1,node),1.f);

  int mult = computeBridgeSilhouetteMultiplicity(bridgeStart,bridgeEnd);

  if(mult!=0)imageAtomicAdd(stencil,ivec2(sampleCoord),mult);
#endif
}


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
shared vec3     localBridgeEnd   [nofLevels][WARP];
#endif
#endif

void debug_storeSilhouetteTraverseStatLastLevel(){
#if STORE_TRAVERSE_STAT == 1
  if(gl_LocalInvocationIndex==0){
    uint w = atomicAdd(debug[0],1);
    debug[1+w*4+0] = job;
    debug[1+w*4+1] = node;
    debug[1+w*4+2] = uint(level);
    debug[1+w*4+3] = 0xff;
  }
#endif
}

void debug_storeSilhouetteTraverseStat(){
#if STORE_TRAVERSE_STAT == 1
  uint w = atomicAdd(debug[0],1);
  debug[1+w*4+0] = job;
  debug[1+w*4+1] = node*WARP + gl_LocalInvocationIndex;
  debug[1+w*4+2] = uint(level);
  debug[1+w*4+3] = status;
#endif
}

void computeBridgeSilhouetteIntersection(in vec3 minCorner,in vec3 aabbSize,int level,uint node){
#if COMPUTE_BRIDGES == 1
  vec4 bridgeStart;
  vec4 bridgeEnd  ;
  int  mult       ;


  bridgeEnd = vec4(minCorner + aabbSize*.5f,1.f);

#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
  localBridgeEnd[level][gl_LocalInvocationIndex] = vec3(bridgeEnd);
#endif

  if(level == 0){
    bridgeStart = clipLightPosition;
  }else{
#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
    bridgeStart = vec4(localBridgeEnd[level-1][node&warpMask],1);
#else
    bridgeStart = vec4(getAABBCenter(level-1,node),1);
#endif
  }

  mult = computeBridgeSilhouetteMultiplicity(bridgeStart,bridgeEnd);
  if(mult!=0)atomicAdd(bridges[nodeLevelOffset[level] + node*WARP + gl_LocalInvocationIndex],mult);
#endif
}

void computeAABBSilhouetteIntersection(out uint status,in vec3 minCorner,in vec3 aabbSize){
  status = TRIVIAL_REJECT;

  vec3 tr;

  tr = trivialRejectCorner3D(edgePlane.xyz);
  if(dot(edgePlane,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  tr = 1.f-tr;
  if(dot(edgePlane,vec4(minCorner + (tr)*(aabbSize),1.f))>0.f)return;

  tr = trivialRejectCorner3D(aPlane.xyz);
  if(dot(aPlane   ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  tr = trivialRejectCorner3D(bPlane.xyz);
  if(dot(bPlane   ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  tr = trivialRejectCorner3D(abPlane.xyz);
  if(dot(abPlane  ,vec4(minCorner + (tr)*(aabbSize),1.f))<0.f)return;

  status = INTERSECTS;
}

void traverseSilhouette(){
  int level = 0;

  uint64_t currentIntersection;

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){

      debug_storeSilhouetteTraverseStatLastLevel();

      lastLevelSilhouette(node);

      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;

        vec3 minCorner;
        vec3 aabbSize;
        getAABB(minCorner,aabbSize,level,(node<<warpBits)+gl_LocalInvocationIndex);
        aabbSize -= minCorner;

        computeBridgeSilhouetteIntersection(minCorner,aabbSize,level,node);

        computeAABBSilhouetteIntersection(status,minCorner,aabbSize);

      }

      debug_storeSilhouetteTraverseStat();

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





//uniform int selectedEdge = -1;

void main(){
  #if STORE_TRAVERSE_STAT != 1
  uint job;
  #endif

  //silhouette loop
  for(;;){
    if(gl_LocalInvocationIndex==0)
      job = atomicAdd(silhouetteJobCounter,1);

    job = readFirstInvocationARB(job);
    if(job >= nofSilhouettes)break;

    //if(selectedEdge>=0 && job != uint(selectedEdge))continue;

    loadSilhouette(job);
    traverseSilhouette();

  }
}



).";

#endif 
