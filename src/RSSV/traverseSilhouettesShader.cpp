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

#define USE_PLANES 1

#define PARALLEL_LOAD 0

shared int  edgeMult;

#if USE_PLANES == 1
uniform mat4 invTran;

#if PARALLEL_LOAD == 1
shared vec4 edgePlane[WARP];
shared vec4 aPlane   [WARP];
shared vec4 bPlane   [WARP];
shared vec4 abPlane  [WARP];
#else
shared vec4 edgePlane;
shared vec4 aPlane;
shared vec4 bPlane;
shared vec4 abPlane;
#endif

#else
shared vec4 edgeA   ;
shared vec4 edgeB   ;
shared vec4 light   ;
#endif

#line 52

#if PARALLEL_LOAD == 1
void parallelLoad(uint job){
  if(job+gl_LocalInvocationIndex >= silhouetteCounter[0])return;

  uint res  = multBuffer[job+gl_LocalInvocationIndex];
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
  
  vec3 n = normalize(cross(edgeB-edgeA,lightPosition.xyz-edgeA));
  edgePlane[gl_LocalInvocationIndex] = invTran*vec4(n,-dot(n,edgeA));

  vec3 an = normalize(cross(n,edgeA-lightPosition.xyz));
  aPlane[gl_LocalInvocationIndex] = invTran*vec4(an,-dot(an,edgeA));

  vec3 bn = normalize(cross(edgeB-lightPosition.xyz,n));
  bPlane[gl_LocalInvocationIndex] = invTran*vec4(bn,-dot(bn,edgeB));

  vec3 abn = normalize(cross(edgeB-edgeA,n));
  abPlane[gl_LocalInvocationIndex] = invTran*vec4(abn,-dot(abn,edgeA));
}
#else
void loadSilhouette(uint job){
  if(gl_LocalInvocationIndex == 0){ // TODO parallel load and precomputation in parallel in separated shader
    uint res  = multBuffer[job];
    uint edge = res & 0x1fffffffu;
    int  mult = int(res) >> 29;

#if USE_PLANES == 1
    vec3 edgeA;
    vec3 edgeB;
    edgeA[0] = edgeBuffer[edge+0*ALIGNED_NOF_EDGES];
    edgeA[1] = edgeBuffer[edge+1*ALIGNED_NOF_EDGES];
    edgeA[2] = edgeBuffer[edge+2*ALIGNED_NOF_EDGES];
    edgeB[0] = edgeBuffer[edge+3*ALIGNED_NOF_EDGES];
    edgeB[1] = edgeBuffer[edge+4*ALIGNED_NOF_EDGES];
    edgeB[2] = edgeBuffer[edge+5*ALIGNED_NOF_EDGES];

    vec3 n = normalize(cross(edgeB-edgeA,lightPosition.xyz-edgeA));
    edgePlane = invTran*vec4(n,-dot(n,edgeA));

    vec3 an = normalize(cross(n,edgeA-lightPosition.xyz));
    aPlane = invTran*vec4(an,-dot(an,edgeA));

    vec3 bn = normalize(cross(edgeB-lightPosition.xyz,n));
    bPlane = invTran*vec4(bn,-dot(bn,edgeB));

    vec3 abn = normalize(cross(edgeB-edgeA,n));
    abPlane = invTran*vec4(abn,-dot(abn,edgeA));
#else
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
    light = proj*view*lightPosition;
#endif

    edgeMult = mult;
  


  }
  memoryBarrierShared();
}

#endif



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

#define USE_SHARED_STACK 1

#if USE_SHARED_STACK == 1
shared uint64_t intersection[nofLevels];
#endif

#if PARALLEL_LOAD == 1
void traverse(uint jb){
#else
void traverse(){
#endif
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
        #if USE_PLANES == 1
                minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 0u]             ;
                maxCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 1u]-minCorner[0];
                minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 2u]             ;
                maxCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 3u]-minCorner[1];
                minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 4u]             ;
                maxCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 5u]-minCorner[2];
        #else
                minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 0u]             ;
                minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 2u]             ;
                minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 4u]             ;
                maxCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 1u]             ;
                maxCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 3u]             ;
                maxCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*WARP*6u + gl_LocalInvocationIndex*6u + 5u]             ;
        #endif
#endif

#if USE_PLANES == 1
        vec3 tr;
        bool planeTest;

#if 1
  #if PARALLEL_LOAD == 1
          status = TRIVIAL_REJECT;
          tr = trivialRejectCorner3D(edgePlane[jb].xyz);
          if(dot(edgePlane[jb],vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f){
            if(dot(edgePlane[jb],vec4(minCorner + (1.f-tr)*(maxCorner),1.f))<=0.f){
              tr = trivialRejectCorner3D(aPlane[jb].xyz);
              if(dot(aPlane[jb],vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f){
                tr = trivialRejectCorner3D(bPlane[jb].xyz);
                if(dot(bPlane[jb],vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f){
                  tr = trivialRejectCorner3D(abPlane[jb].xyz);
                  if(dot(abPlane[jb],vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f)
                    status = INTERSECTS;
                }
              }
            }
          }
  #else
          status = TRIVIAL_REJECT;
          tr = trivialRejectCorner3D(edgePlane.xyz);
          if(dot(edgePlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f){
            if(dot(edgePlane,vec4(minCorner + (1.f-tr)*(maxCorner),1.f))<=0.f){
              tr = trivialRejectCorner3D(aPlane.xyz);
              if(dot(aPlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f){
                tr = trivialRejectCorner3D(bPlane.xyz);
                if(dot(bPlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f){
                  tr = trivialRejectCorner3D(abPlane.xyz);
                  if(dot(abPlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f)
                    status = INTERSECTS;
                }
              }
            }
          }
  #endif
#endif

#if 0
        tr = trivialRejectCorner3D(edgePlane.xyz);
        planeTest =              dot(edgePlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f;
        planeTest = planeTest && dot(edgePlane,vec4(minCorner + (1.f-tr)*(maxCorner),1.f))<=0.f;
        tr = trivialRejectCorner3D(aPlane.xyz);
        planeTest = planeTest && dot(aPlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f;
        tr = trivialRejectCorner3D(bPlane.xyz);
        planeTest = planeTest && dot(bPlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f;
        tr = trivialRejectCorner3D(abPlane.xyz);
        planeTest = planeTest && dot(abPlane,vec4(minCorner + (    tr)*(maxCorner),1.f))>=0.f;

        if(planeTest)
          status = INTERSECTS;
        else
          status = TRIVIAL_REJECT;
#endif
#else
        status = silhouetteStatus(edgeA,edgeB,light,minCorner,maxCorner);
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
  #if STORE_TRAVERSE_STAT == 1
  #else
  uint job;
  #endif

  for(;;){
    if(gl_LocalInvocationIndex==0){
#if PARALLEL_LOAD == 1
      job = atomicAdd(jobCounter[0],WARP);
#else
      job = atomicAdd(jobCounter[0],1);
#endif
    }

    job = readFirstInvocationARB(job);
#if PARALLEL_LOAD == 1
    parallelLoad(job);

    if(job >= silhouetteCounter[0])return;traverse( 0);job++;
    if(job >= silhouetteCounter[0])return;traverse( 1);job++;
    if(job >= silhouetteCounter[0])return;traverse( 2);job++;
    if(job >= silhouetteCounter[0])return;traverse( 3);job++;
    if(job >= silhouetteCounter[0])return;traverse( 4);job++;
    if(job >= silhouetteCounter[0])return;traverse( 5);job++;
    if(job >= silhouetteCounter[0])return;traverse( 6);job++;
    if(job >= silhouetteCounter[0])return;traverse( 7);job++;
    if(job >= silhouetteCounter[0])return;traverse( 8);job++;
    if(job >= silhouetteCounter[0])return;traverse( 9);job++;
    if(job >= silhouetteCounter[0])return;traverse(10);job++;
    if(job >= silhouetteCounter[0])return;traverse(11);job++;
    if(job >= silhouetteCounter[0])return;traverse(12);job++;
    if(job >= silhouetteCounter[0])return;traverse(13);job++;
    if(job >= silhouetteCounter[0])return;traverse(14);job++;
    if(job >= silhouetteCounter[0])return;traverse(15);job++;
    if(job >= silhouetteCounter[0])return;traverse(16);job++;
    if(job >= silhouetteCounter[0])return;traverse(17);job++;
    if(job >= silhouetteCounter[0])return;traverse(18);job++;
    if(job >= silhouetteCounter[0])return;traverse(19);job++;
    if(job >= silhouetteCounter[0])return;traverse(20);job++;
    if(job >= silhouetteCounter[0])return;traverse(21);job++;
    if(job >= silhouetteCounter[0])return;traverse(22);job++;
    if(job >= silhouetteCounter[0])return;traverse(23);job++;
    if(job >= silhouetteCounter[0])return;traverse(24);job++;
    if(job >= silhouetteCounter[0])return;traverse(25);job++;
    if(job >= silhouetteCounter[0])return;traverse(26);job++;
    if(job >= silhouetteCounter[0])return;traverse(27);job++;
    if(job >= silhouetteCounter[0])return;traverse(28);job++;
    if(job >= silhouetteCounter[0])return;traverse(29);job++;
    if(job >= silhouetteCounter[0])return;traverse(30);job++;
    if(job >= silhouetteCounter[0])return;traverse(31);job++;
    if(job >= silhouetteCounter[0])return;traverse(32);job++;
    if(job >= silhouetteCounter[0])return;traverse(33);job++;
    if(job >= silhouetteCounter[0])return;traverse(34);job++;
    if(job >= silhouetteCounter[0])return;traverse(35);job++;
    if(job >= silhouetteCounter[0])return;traverse(36);job++;
    if(job >= silhouetteCounter[0])return;traverse(37);job++;
    if(job >= silhouetteCounter[0])return;traverse(38);job++;
    if(job >= silhouetteCounter[0])return;traverse(39);job++;
    if(job >= silhouetteCounter[0])return;traverse(40);job++;
    if(job >= silhouetteCounter[0])return;traverse(41);job++;
    if(job >= silhouetteCounter[0])return;traverse(42);job++;
    if(job >= silhouetteCounter[0])return;traverse(43);job++;
    if(job >= silhouetteCounter[0])return;traverse(44);job++;
    if(job >= silhouetteCounter[0])return;traverse(45);job++;
    if(job >= silhouetteCounter[0])return;traverse(46);job++;
    if(job >= silhouetteCounter[0])return;traverse(47);job++;
    if(job >= silhouetteCounter[0])return;traverse(48);job++;
    if(job >= silhouetteCounter[0])return;traverse(49);job++;
    if(job >= silhouetteCounter[0])return;traverse(50);job++;
    if(job >= silhouetteCounter[0])return;traverse(51);job++;
    if(job >= silhouetteCounter[0])return;traverse(52);job++;
    if(job >= silhouetteCounter[0])return;traverse(53);job++;
    if(job >= silhouetteCounter[0])return;traverse(54);job++;
    if(job >= silhouetteCounter[0])return;traverse(55);job++;
    if(job >= silhouetteCounter[0])return;traverse(56);job++;
    if(job >= silhouetteCounter[0])return;traverse(57);job++;
    if(job >= silhouetteCounter[0])return;traverse(58);job++;
    if(job >= silhouetteCounter[0])return;traverse(59);job++;
    if(job >= silhouetteCounter[0])return;traverse(60);job++;
    if(job >= silhouetteCounter[0])return;traverse(61);job++;
    if(job >= silhouetteCounter[0])return;traverse(62);job++;
    if(job >= silhouetteCounter[0])return;traverse(63);job++;
                                                   
#else
    if(job >= silhouetteCounter[0])return;

    loadSilhouette(job);

    traverse();
#endif

  }
}
).";
