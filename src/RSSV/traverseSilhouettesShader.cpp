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

#ifndef MORE_PLANES
#define MORE_PLANES 0
#endif//MORE_PLANES

#ifndef ENABLE_FFC
#define ENABLE_FFC 0
#endif//ENABLE_FFC

#ifndef NO_AABB
#define NO_AABB 0
#endif//NO_AABB

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer NodePool          {uint  nodePool         [];};
layout(std430,binding=1)buffer AABBPool          {float aabbPool         [];};
layout(std430,binding=2)buffer JobCounter        {uint  jobCounter       [];};
layout(std430,binding=3)buffer EdgeBuffer        {float edgeBuffer       [];};
layout(std430,binding=4)buffer MultBuffer        {uint  multBuffer       [];};
layout(std430,binding=5)buffer SilhouetteCounter {uint  silhouetteCounter[];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;

uniform mat4 view;
uniform mat4 proj;
uniform vec4 lightPosition;

shared vec4 edgeA   ;
shared vec4 edgeB   ;
shared vec4 light   ;
shared int  edgeMult;

#line 52

shared uint ee;

void loadSilhouette(uint job){
  if(gl_LocalInvocationIndex == 0){ // TODO parallel load and precomputation in parallel in separated shader
    uint res  = multBuffer[job];
    uint edge = res & 0x1fffffffu;
    int  mult = int(res) >> 29;
    //ee = edge;
    //edge = job;
    vec4 point;

    point[0] = edgeBuffer[edge+0*ALIGNED_NOF_EDGES];
    point[1] = edgeBuffer[edge+1*ALIGNED_NOF_EDGES];
    point[2] = edgeBuffer[edge+2*ALIGNED_NOF_EDGES];
    point[3] = 1;
    edgeA = proj*view*point;

    point[0] = edgeBuffer[edge+3*ALIGNED_NOF_EDGES];
    point[1] = edgeBuffer[edge+4*ALIGNED_NOF_EDGES];
    point[2] = edgeBuffer[edge+5*ALIGNED_NOF_EDGES];
    point[3] = 1;
    edgeB = proj*view*point;

    edgeMult = mult;

    light = proj*view*lightPosition;
  }
  memoryBarrierShared();
}

#if STORE_TRAVERSE_STAT == 1
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif

vec3 trivialRejectCorner3D(vec3 Normal){
  return vec3((ivec3(sign(Normal))+1)/2);
}

uint job = 0u;

#if WARP == 64
#line 10000
void traverse(){
  int level = 0;
  uint64_t intersection[nofLevels];

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
        vec3 maxCorner;
        minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 0u]             ;
        minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 2u]             ;
        minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 4u]             ;
        maxCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 1u]             ;
        maxCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 3u]             ;
        maxCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 5u]             ;
        //aabbSize [0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 1u]-minCorner[0];
        //aabbSize [1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 3u]-minCorner[1];
        //aabbSize [2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 5u]-minCorner[2];
#endif


        status = silhouetteStatus(edgeA,edgeB,light,minCorner,maxCorner);

      }

#if STORE_TRAVERSE_STAT == 1
        uint w = atomicAdd(debug[0],1);
        debug[1+w*4+0] = job;
        debug[1+w*4+1] = node*WARP + gl_LocalInvocationIndex;
        debug[1+w*4+2] = uint(level);
        debug[1+w*4+3] = status;
#endif

      intersection[level] = ballotARB(status == INTERSECTS    );
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
    //*
  for(;;){
    if(gl_LocalInvocationIndex==0){
      job = atomicAdd(jobCounter[0],1);
    }

    job = readFirstInvocationARB(job);
    if(job >= silhouetteCounter[0])return;
    //if(job >= 6)return;

    loadSilhouette(job);

    traverse();

  }
  // */
  //for(;job<6;++job){
  //  loadSilhouette(job);
  //  traverse();
  //}
  //job=3;loadSilhouette(job);traverse();
  //job=1;loadSilhouette(job);traverse();
  //job=5;loadSilhouette(job);traverse();
  //job=2;loadSilhouette(job);traverse();
  //job=0;loadSilhouette(job);traverse();
  //job=4;loadSilhouette(job);traverse();
}
).";
