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

const uint planesPerSF = 4u + MORE_PLANES*3u;
const uint floatsPerPlane = 4u;
const uint floatsPerSF = planesPerSF * floatsPerPlane;

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer NodePool    {uint  nodePool    [];};
layout(std430,binding=1)buffer AABBPool    {float aabbPool    [];};
layout(std430,binding=2)buffer ShadowFrusta{float shadowFrusta[];};
layout(std430,binding=3)buffer JobCounter  {uint  jobCounter  [];};

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
}


vec3 trivialRejectCorner3D(vec3 Normal){
  return vec3((ivec3(sign(Normal))+1)/2);
}

const uint TRIVIAL_REJECT = 0xf0u;
const uint TRIVIAL_ACCEPT =    3u;
const uint INTERSECTS     =    2u;

//layout(std430,binding=6)buffer Deb{float deb[];};
//layout(std430,binding=7)buffer Debc{uint debc[];};

/*
void debugStoreAABB(vec3 minCorner,vec3 size,vec4 plane){
  return;
  uint w = atomicAdd(debc[0],1);
  uint NN = 17;
  deb[w*NN+0] = minCorner[0];
  deb[w*NN+1] = minCorner[1];
  deb[w*NN+2] = minCorner[2];
  deb[w*NN+3] = size[0];
  deb[w*NN+4] = size[1];
  deb[w*NN+5] = size[2];
  deb[w*NN+6] = plane[0];
  deb[w*NN+7] = plane[1];
  deb[w*NN+8] = plane[2];
  deb[w*NN+9] = plane[3];
  vec3 trr;
  trr = minCorner + trivialRejectCorner3D(plane.xyz)*size;
  deb[w*NN+10] = trr[0];
  deb[w*NN+11] = trr[1];
  deb[w*NN+12] = trr[2];
  trr = minCorner + (1.f-trivialRejectCorner3D(plane.xyz))*size;
  deb[w*NN+13] = trr[0];
  deb[w*NN+14] = trr[1];
  deb[w*NN+15] = trr[2];
  deb[w*NN+16] = float(gl_LocalInvocationIndex);
}

// */

uint trivialRejectAccept(vec3 minCorner,vec3 size){
  uint status = TRIVIAL_ACCEPT;
  vec4 plane;
  vec3 tr;

  plane = vec4(shadowFrustaPlanes[0],shadowFrustaPlanes[1],shadowFrustaPlanes[2],shadowFrustaPlanes[3]);
  //debugStoreAABB(minCorner,size,plane);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[4],shadowFrustaPlanes[5],shadowFrustaPlanes[6],shadowFrustaPlanes[7]);
  //debugStoreAABB(minCorner,size,plane);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[8],shadowFrustaPlanes[9],shadowFrustaPlanes[10],shadowFrustaPlanes[11]);
  //debugStoreAABB(minCorner,size,plane);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[12],shadowFrustaPlanes[13],shadowFrustaPlanes[14],shadowFrustaPlanes[15]);
  //debugStoreAABB(minCorner,size,plane);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
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
  plane;

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
  plane;

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

layout(std430,binding = 7)buffer Debug{uint debug[];};

#if WARP == 32

void traverse(){
  int level = 0;
  uint intersection[nofLevels];
  //uvec2 intersection[nofLevels];
  //if(nofLevels > 0)intersection[0] = uvec2(0);
  //if(nofLevels > 1)intersection[1] = uvec2(0);
  //if(nofLevels > 2)intersection[2] = uvec2(0);
  //if(nofLevels > 3)intersection[3] = uvec2(0);

  uint node = 0;
  while(level >= 0){
    //if(gl_LocalInvocationIndex==0){
    //  uint w = atomicAdd(debc[0],1);
    //  if(w<100){
    //    debc[w*3+1+0] = uint(level);
    //    debc[w*3+1+1] = uint(intersection[0]);
    //    debc[w*3+1+2] = uint(intersection[1]);
    //  }
    //}

    if(level == int(nofLevels)){
        if(level >  int(nofLevels))return;
        if(gl_LocalInvocationIndex==0){
          uint w = atomicAdd(debug[0],1);
          debug[1+w*3+0] = job;
          debug[1+w*3+1] = node;
          debug[1+w*3+2] = uint(level);
        }

      //test pixels
#if 1
      lastLevel(node);
#endif
      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node]&uint(1u<<(gl_LocalInvocationIndex)));
      if(status != 0u){
        if(level >  int(nofLevels))return;

        //if(gl_LocalInvocationIndex == 0)
        {
          uint w = atomicAdd(debug[0],1);
          debug[1+w*3+0] = job;
          debug[1+w*3+1] = node;
          debug[1+w*3+2] = uint(level);
        }

        vec3 minCorner;
        vec3 aabbSize;
        minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 0u]             ;
        minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 2u]             ;
        minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 4u]             ;
        aabbSize [0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 1u]-minCorner[0];
        aabbSize [1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 3u]-minCorner[1];
        aabbSize [2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 5u]-minCorner[2];


        status = trivialRejectAccept(minCorner,aabbSize);
      }
      intersection[level] = unpackUint2x32(ballotARB(status == INTERSECTS    ))[0];
      uint     trA        = unpackUint2x32(ballotARB(status == TRIVIAL_ACCEPT))[0];

#if 1
      if(gl_LocalInvocationIndex == 0u){
        if(trA != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node],~trA);
      }
#endif

    }

    while(level >= 0 && intersection[level] == 0u){
      node >>= warpBits;
      level--;
    }
    if(level < 0)break;
    if(level>=0){
      uint selectedBit = findLSB(intersection[level]);
      node <<= warpBits   ;
      node  += selectedBit;

      intersection[level] ^= 1u << selectedBit;

      level++;
    }
  }
}

#endif

#if WARP == 64


void traverse(){
  int level = 0;
  uint64_t intersection[nofLevels];
  //uvec2 intersection[nofLevels];
  //if(nofLevels > 0)intersection[0] = uvec2(0);
  //if(nofLevels > 1)intersection[1] = uvec2(0);
  //if(nofLevels > 2)intersection[2] = uvec2(0);
  //if(nofLevels > 3)intersection[3] = uvec2(0);

  uint node = 0;
  while(level >= 0){
    //if(gl_LocalInvocationIndex==0){
    //  uint w = atomicAdd(debc[0],1);
    //  if(w<100){
    //    debc[w*3+1+0] = uint(level);
    //    debc[w*3+1+1] = uint(intersection[0]);
    //    debc[w*3+1+2] = uint(intersection[1]);
    //  }
    //}
    if(level == int(nofLevels)){
      if(level >  int(nofLevels))return;
      if(gl_LocalInvocationIndex==0){
        uint w = atomicAdd(debug[0],1);
        debug[1+w*3+0] = job;
        debug[1+w*3+1] = node;
        debug[1+w*3+2] = uint(level);
      }
      //test pixels
#if 1
      lastLevel(node);
#endif
      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;
        uint w = atomicAdd(debug[0],1);
        debug[1+w*3+0] = job;
        debug[1+w*3+1] = node;
        debug[1+w*3+2] = uint(level);

        vec3 minCorner;
        vec3 aabbSize;
        minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 0u]             ;
        minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 2u]             ;
        minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 4u]             ;
        aabbSize [0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 1u]-minCorner[0];
        aabbSize [1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 3u]-minCorner[1];
        aabbSize [2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 5u]-minCorner[2];


        status = trivialRejectAccept(minCorner,aabbSize);
      }
      intersection[level] = ballotARB(status == INTERSECTS    );
      uint64_t trA        = ballotARB(status == TRIVIAL_ACCEPT);

#if 1
      if(gl_LocalInvocationIndex == 0u){
        if(unpackUint2x32(trA)[0] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+0],~unpackUint2x32(trA)[0]);
        if(unpackUint2x32(trA)[1] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+1],~unpackUint2x32(trA)[1]);
      }
#endif

    }

    while(level >= 0 && intersection[level] == 0ul){
      node >>= warpBits;
      level--;
    }
    if(level < 0)break;
    if(level>=0){
      uint selectedBit = unpackUint2x32(intersection[level])[0]!=0?findLSB(unpackUint2x32(intersection[level])[0]):findLSB(unpackUint2x32(intersection[level])[1])+32u;
      node <<= warpBits   ;
      node  += selectedBit;

      uint64_t mask = 1ul;
      mask <<= selectedBit;
      intersection[level] ^= mask;

      level++;
    }
  }
}

#endif

#line 36
void main(){
  //uint job = 0u;
  for(;;){
    if(gl_LocalInvocationIndex==0){
      job = atomicAdd(jobCounter[0],1);
    }

    job = readFirstInvocationARB(job);
    if(job >= NOF_TRIANGLES)return;

    loadShadowFrustum(job);

    //uint w = atomicAdd(debug[0],1);
    //debug[1+w*3+0] = job;
    //debug[1+w*3+1] = 0;
    //debug[1+w*3+2] = uint(0);

    traverse();

  }
}

).";
