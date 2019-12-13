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

#define PLANES_PER_SF 4

const uint floatsPerPlane = 4u;
const uint floatsPerSF    = floatsPerPlane * PLANES_PER_SF;

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

const uint TRIVIAL_REJECT = 4u;
const uint TRIVIAL_ACCEPT = 3u;
const uint INTERSECTS     = 2u;

layout(std430,binding=6)buffer Deb{float deb[];};
layout(std430,binding=7)buffer Debc{uint debc[];};

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

  return status;
}

/*
#define TEST_SHADOW_FRUSTUM_LAST(LEVEL)\
void JOIN(TestShadowFrustumHDB,LEVEL)(uvec2 coord,vec2 clipCoord){\
  uvec2 localCoord             = getLocalCoords(LEVEL);\
  uvec2 globalCoord            = coord * JOIN(TILE_DIVISIBILITY,LEVEL) + localCoord;\
  vec2  globalClipCoord        = clipCoord + JOIN(TILE_SIZE_IN_CLIP_SPACE,LEVEL) * localCoord;\
  if(texelFetch(triangleID,ivec2(globalCoord),0).x == SHADOWFRUSTUM_ID_IN_DISPATCH)return;\
  vec4  SampleCoordInClipSpace = vec4(\
    globalClipCoord + JOIN(TILE_SIZE_IN_CLIP_SPACE,LEVEL)*.5,\
    texelFetch(HDT[LEVEL],ivec2(globalCoord),0).x,1);\
  if(SampleCoordInClipSpace.z >= 1)return;\
  bool inside = true;\
  for(int p = 0; p < NOF_PLANES_PER_SF; ++p)\
    inside=inside && dot(SampleCoordInClipSpace,SharedShadowFrusta[SHADOWFRUSTUM_ID_IN_WORKGROUP][p])>=0;\
  if(inside)\
    imageStore(FinalStencilMask,ivec2(globalCoord),uvec4(SHADOW_VALUE));\
}

*/
//imageStore(shadowMask,ivec2(gl_GlobalInvocationID.xy),vec4(1-S));

void traverse(){
  int level = 0;
  uint64_t intersection[nofLevels];

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){
      //test pixels
      //uvec2 tileCoord = demorton(node).xy;
      //imageStore(shadowMask,ivec2(0,0) + ivec2(gl_LocalInvocationIndex%8,gl_LocalInvocationIndex/8),vec4(0));
      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){
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

      //imageStore(shadowMask,ivec2(64+level*16,64) + ivec2(gl_LocalInvocationIndex%8,gl_LocalInvocationIndex/8),vec4(0));
      //if(status == INTERSECTS)
      //  imageStore(shadowMask,ivec2(64+level*16,64) + ivec2(gl_LocalInvocationIndex%8,gl_LocalInvocationIndex/8),vec4(0));

      if(gl_LocalInvocationIndex == 0u){
        if(unpackUint2x32(trA)[0] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+0],~unpackUint2x32(trA)[0]);
        if(unpackUint2x32(trA)[1] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+1],~unpackUint2x32(trA)[1]);
      }

      //if(intersection[level] != 0)
      //  imageStore(shadowMask,ivec2(256,256) + ivec2(gl_LocalInvocationIndex%8,gl_LocalInvocationIndex/8),vec4(0));
      //if(trA != 0)
      //  imageStore(shadowMask,ivec2(128,128) + ivec2(gl_LocalInvocationIndex%8,gl_LocalInvocationIndex/8),vec4(0));
      //if(trA != 0)
      //  imageStore(shadowMask,ivec2(128,64) + ivec2(gl_LocalInvocationIndex%8,gl_LocalInvocationIndex/8),vec4(0));
      //if(trA != 0)
      //  imageStore(shadowMask,ivec2(128,64) + ivec2(gl_LocalInvocationIndex%8,gl_LocalInvocationIndex/8),vec4(0));
    }

    if(intersection[level] == 0ul){
      node >>= warpBits;
      level--;
    }else{
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


#line 36
void main(){
  uint job;
  for(;;){
    if(gl_LocalInvocationIndex==0){
      job = atomicAdd(jobCounter[0],1);
    }

    job = readFirstInvocationARB(job);
    if(job >= NOF_TRIANGLES)return;

    loadShadowFrustum(job);

    //{
    //  uint xx = job % 100;
    //  uint yy = job / 100;
    //  imageStore(shadowMask,ivec2(xx,yy),vec4(0));
    //}
    traverse();

  }
}

).";
