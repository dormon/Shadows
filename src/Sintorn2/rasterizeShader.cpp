#include <Sintorn2/rasterizeShader.h>

std::string const sintorn2::rasterizeShader = R".(

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

#ifndef MORE_PLANES
#define MORE_PLANES 0
#endif//MORE_PLANES

#ifndef ENABLE_FFC
#define ENABLE_FFC 0
#endif//ENABLE_FFC

#ifndef NO_AABB
#define NO_AABB 0
#endif//NO_AABB

const uint planesPerSF = 4u + MORE_PLANES*3u;
const uint floatsPerPlane = 4u;
const uint floatsPerSF = planesPerSF * floatsPerPlane + uint(ENABLE_FFC);

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer NodePool    {uint  nodePool    [];};
layout(std430,binding=1)buffer AABBPool    {float aabbPool    [];};
layout(std430,binding=2)buffer ShadowFrusta{float shadowFrusta[];};
layout(std430,binding=3)buffer JobCounter  {uint  jobCounter  [];};

#if MEMORY_OPTIM == 1
layout(std430,binding=5)buffer AABBPointer {uint  aabbPointer [];};
#endif

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;


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

#if SAVE_COLLISION == 1
layout(std430,binding=5)buffer Deb{float deb[];};
layout(std430,binding=6)buffer Debc{uint debc[];};
#endif

#if STORE_TRAVERSE_STAT == 1
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif



vec3 trivialRejectCorner3D(vec3 Normal){
  return vec3((ivec3(sign(Normal))+1)/2);
}

const uint TRIVIAL_REJECT = 0xf0u;
const uint TRIVIAL_ACCEPT =    3u;
const uint INTERSECTS     =    2u;

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
#if TRIANGLE_INTERSECT == 1
  if(dot(plane,vec4(minCorner + tr*size,1.f))>0.f)
    return TRIVIAL_REJECT;
#endif
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1.f),plane)>0.f);

#if MORE_PLANES == 1
  if(status == INTERSECTS){
    plane = vec4(shadowFrustaPlanes[16],shadowFrustaPlanes[17],shadowFrustaPlanes[18],shadowFrustaPlanes[19]);
    tr    = trivialRejectCorner3D(plane.xyz);
    if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
      return TRIVIAL_REJECT;

    plane = vec4(shadowFrustaPlanes[20],shadowFrustaPlanes[21],shadowFrustaPlanes[22],shadowFrustaPlanes[23]);
    tr    = trivialRejectCorner3D(plane.xyz);
    if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
      return TRIVIAL_REJECT;

    plane = vec4(shadowFrustaPlanes[24],shadowFrustaPlanes[25],shadowFrustaPlanes[26],shadowFrustaPlanes[27]);
    tr    = trivialRejectCorner3D(plane.xyz);
    if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
      return TRIVIAL_REJECT;
  }
#endif

  return status;
}


void lastLevel(uint node){
  uvec2 sampleCoord;
  vec4 clipCoord;
  bool inside;
  vec4 plane;


  sampleCoord = (demorton(node).xy<<uvec2(tileBitsX,tileBitsY)) + uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);

  clipCoord.z = texelFetch(depthTexture,ivec2(sampleCoord)).x*2-1;
  clipCoord.xy = -1+2*((vec2(sampleCoord) + vec2(0.5)) / vec2(WINDOW_X,WINDOW_Y));
  clipCoord.w = 1.f;

  inside = true;

  plane = vec4(shadowFrustaPlanes[0],shadowFrustaPlanes[1],shadowFrustaPlanes[2],shadowFrustaPlanes[3]);
  inside = inside && (dot(plane,clipCoord) >= 0);
  plane = vec4(shadowFrustaPlanes[4],shadowFrustaPlanes[5],shadowFrustaPlanes[6],shadowFrustaPlanes[7]);
  inside = inside && (dot(plane,clipCoord) >= 0);
  plane = vec4(shadowFrustaPlanes[8],shadowFrustaPlanes[9],shadowFrustaPlanes[10],shadowFrustaPlanes[11]);
  inside = inside && (dot(plane,clipCoord) >= 0);
  plane = vec4(shadowFrustaPlanes[12],shadowFrustaPlanes[13],shadowFrustaPlanes[14],shadowFrustaPlanes[15]);
  inside = inside && (dot(plane,clipCoord) >= 0);

  if(inside)
    imageStore(shadowMask,ivec2(sampleCoord),vec4(0));



  sampleCoord = (demorton(node).xy<<uvec2(tileBitsX,tileBitsY)) + uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX) + uvec2(0u,4u);

  clipCoord.z = texelFetch(depthTexture,ivec2(sampleCoord)).x*2-1;
  clipCoord.xy = -1+2*((vec2(sampleCoord) + vec2(0.5)) / vec2(WINDOW_X,WINDOW_Y));
  clipCoord.w = 1.f;

  inside = true;

  plane = vec4(shadowFrustaPlanes[0],shadowFrustaPlanes[1],shadowFrustaPlanes[2],shadowFrustaPlanes[3]);
  inside = inside && (dot(plane,clipCoord) >= 0);
  plane = vec4(shadowFrustaPlanes[4],shadowFrustaPlanes[5],shadowFrustaPlanes[6],shadowFrustaPlanes[7]);
  inside = inside && (dot(plane,clipCoord) >= 0);
  plane = vec4(shadowFrustaPlanes[8],shadowFrustaPlanes[9],shadowFrustaPlanes[10],shadowFrustaPlanes[11]);
  inside = inside && (dot(plane,clipCoord) >= 0);
  plane = vec4(shadowFrustaPlanes[12],shadowFrustaPlanes[13],shadowFrustaPlanes[14],shadowFrustaPlanes[15]);
  inside = inside && (dot(plane,clipCoord) >= 0);

  if(inside)
    imageStore(shadowMask,ivec2(sampleCoord),vec4(0));
}


uint job = 0u;

#define USE_SHARED_STACK 1

#if WARP == 32

#if USE_SHARED_STACK == 1
shared uint intersection[nofLevels];
#endif

void traverse(){
  int level = 0;

#if USE_SHARED_STACK == 1
  uint currentIntersection;
#else
  uint intersection[nofLevels];
#endif

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

#if COMPUTE_LAST_LEVEL == 1
      lastLevel(node);
#endif
      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node]&uint(1u<<(gl_LocalInvocationIndex)));
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

#if STORE_TRAVERSE_STAT == 1
        uint w = atomicAdd(debug[0],1);
        debug[1+w*4+0] = job;
        debug[1+w*4+1] = node;
        debug[1+w*4+2] = uint(level);
        debug[1+w*4+3] = status;
#endif

#if USE_SHARED_STACK == 1
      currentIntersection = unpackUint2x32(ballotARB(status == INTERSECTS    ))[0];
      if(gl_LocalInvocationIndex==0)
        intersection[level] = currentIntersection;
      memoryBarrierShared();
#else
      intersection[level] = unpackUint2x32(ballotARB(status == INTERSECTS    ))[0];
#endif


#if USE_TA_OPTIM == 1
      uint     trA        = unpackUint2x32(ballotARB(status == TRIVIAL_ACCEPT))[0];
      if(gl_LocalInvocationIndex == 0u){
        if(trA != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node],~trA);
      }
#endif

    }

#if USE_SHARED_STACK == 1
    while(level >= 0 && currentIntersection == 0u){
      node >>= warpBits;
      level--;
      if(level < 0)break;
      currentIntersection = intersection[level];
    }
#else
    while(level >= 0 && intersection[level] == 0u){
      node >>= warpBits;
      level--;
    }
#endif



    if(level < 0)break;
    if(level>=0){
#if USE_SHARED_STACK == 1
      uint selectedBit = findLSB(currentIntersection);
#else
      uint selectedBit = findLSB(intersection[level]);
#endif
      node <<= warpBits   ;
      node  += selectedBit;

      uint mask = 1u;
      mask <<= selectedBit;

#if USE_SHARED_STACK == 1
      currentIntersection ^= mask;
      if(gl_LocalInvocationIndex==0)
        intersection[level] = currentIntersection;
      memoryBarrierShared();
#else
      intersection[level] ^= mask;
#endif


      level++;
    }
  }
}

#endif

#if WARP == 64


#if USE_SHARED_STACK == 1
shared uint64_t intersection[nofLevels];
#endif

void traverse(){
  int level = 0;

#if USE_SHARED_STACK == 1
  uint64_t currentIntersection;
#else
  uint64_t intersection[nofLevels];
#endif

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

#if COMPUTE_LAST_LEVEL == 1
      lastLevel(node);
#endif
      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;

        vec3 minCorner;
        vec3 aabbSize;
#if NO_AABB == 1
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


        status = trivialRejectAccept(minCorner,aabbSize);

#if SAVE_COLLISION == 1
#line 363
        uint w = atomicAdd(debc[0],1);
        if(w<10000){
          const uint NN = 1+1+3+3+floatsPerSF;
          deb[w*NN+0] = float(status);
          deb[w*NN+1] = float(level);

          for(uint i=0;i<3;++i)
            deb[w*NN+1+1+i] = 
              //1.f+float(i);
              minCorner[i];

          for(uint i=0;i<3;++i)
            deb[w*NN+1+1+3+i] = 
              //1.f+3.f+float(i);
              aabbSize[i];

          for(uint i=0;i<floatsPerSF;++i)
            deb[w*NN+1+1+3+3+i] = 
              //1.f+3.f+3.f+float(i);
              shadowFrustaPlanes[i];
        }
#endif



      }

#if STORE_TRAVERSE_STAT == 1
        uint w = atomicAdd(debug[0],1);
        debug[1+w*4+0] = job;
        debug[1+w*4+1] = node*WARP + gl_LocalInvocationIndex;
        debug[1+w*4+2] = uint(level);
        debug[1+w*4+3] = status;
#endif

#if USE_SHARED_STACK == 1
      currentIntersection = ballotARB(status == INTERSECTS    );
      if(gl_LocalInvocationIndex==0)
        intersection[level] = currentIntersection;
#else
      intersection[level] = ballotARB(status == INTERSECTS    );
#endif


#if USE_TA_OPTIM == 1
      uint64_t trA        = ballotARB(status == TRIVIAL_ACCEPT);
      if(gl_LocalInvocationIndex == 0u){
        if(unpackUint2x32(trA)[0] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+0],~unpackUint2x32(trA)[0]);
        if(unpackUint2x32(trA)[1] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+1],~unpackUint2x32(trA)[1]);
      }
#endif

    }

#if USE_SHARED_STACK == 1
    while(level >= 0 && currentIntersection == 0ul){
      node >>= warpBits;
      level--;
      if(level < 0)break;
      currentIntersection = intersection[level];
    }
#else
    while(level >= 0 && intersection[level] == 0ul){
      node >>= warpBits;
      level--;
    }
#endif


    if(level < 0)break;
    if(level>=0){
#if USE_SHARED_STACK == 1
      uint selectedBit = unpackUint2x32(currentIntersection)[0]!=0?findLSB(unpackUint2x32(currentIntersection)[0]):findLSB(unpackUint2x32(currentIntersection)[1])+32u;
#else
      uint selectedBit = unpackUint2x32(intersection[level])[0]!=0?findLSB(unpackUint2x32(intersection[level])[0]):findLSB(unpackUint2x32(intersection[level])[1])+32u;
#endif


      node <<= warpBits   ;
      node  += selectedBit;

      uint64_t mask = 1ul;
      mask <<= selectedBit;

#if USE_SHARED_STACK == 1
      currentIntersection ^= mask;
      if(gl_LocalInvocationIndex==0)
        intersection[level] = currentIntersection;
#else
      intersection[level] ^= mask;
#endif

      level++;
    }
  }
}

#endif

#line 36
void main(){
  //if(texelFetch(depthTexture,ivec2(0,0)).x != 1337)return;
  for(;;){
    if(gl_LocalInvocationIndex==0){
      job = atomicAdd(jobCounter[0],1);
    }

    job = readFirstInvocationARB(job);
    if(job >= NOF_TRIANGLES)return;

    loadShadowFrustum(job);

    #if (ENABLE_FFC == 1)
      if(shadowFrustaPlanes[floatsPerSF-1] == 1.f)
        continue;
    #endif

    traverse();

  }
}

).";
