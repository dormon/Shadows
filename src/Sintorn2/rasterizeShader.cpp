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

const uint floatsPerPlane = 4;
const uint floatsPerSF    = floatsPerPlane * PLANES_PER_SF;

layout(local_size_x=WARP)in;

layout(std430,binding=0)buffer NodePool    {uint  nodePool    [];};
layout(std430,binding=1)buffer AABBPool    {float aabbPool    [];};
layout(std430,binding=2)buffer ShadowFrusta{float shadowFrusta[];};
layout(std430,binding=3)buffer JobCounter  {uint  jobCounter  [];};

layout(     binding=0)          uniform sampler2DRect depthTexture;
layout(r32f,binding=1)writeonly uniform image2D       shadowMask  ;


const uint alignedNofSF = (uint(NOF_TRIANGLES / SF_ALIGNMENT) + uint((NOF_TRIANGLES % SF_ALIGNMENT) != 0u)) * SF_ALIGNMENT;

void loadShadowFrustum(uint job){
  if(gl_LocalInvocationIndex < floatsPerSF){
#if TRIANGLE_INTERLEAVE == 1
    shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[alignedNofSF*gl_LocalInvocationIndex + job];
#else
    shadowFrustaPlanes[gl_LocalInvocationIndex] = shadowFrusta[job*floatsPerSF+gl_LocalInvocationIndex];
#endif
  }
}

shared float shadowFrustaPlanes[floatsPerSF];

vec3 trivialRejectCorner3D(vec3 Normal){
  return vec3((ivec3(sign(Normal))+1)/2);
}

const uint TRIVIAL_REJECT = 4u;
const uint TRIVIAL_ACCEPT = 3u;
const uint INTERSECTS     = 2u;

uint trivialRejectAccept(vec3 minCorner,vec3 size){
  uint status = TRIVIAL_ACCEPT;
  vec4 plane;
  vec3 tr;

  plane = vec4(shadowFrustaPlanes[0],shadowFrustaPlanes[1],shadowFrustaPlanes[2],shadowFrustaPlanes[3]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
  status &= uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[4],shadowFrustaPlanes[5],shadowFrustaPlanes[6],shadowFrustaPlanes[7]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
  status &= uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[8],shadowFrustaPlanes[9],shadowFrustaPlanes[10],shadowFrustaPlanes[11]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
  status &= uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = vec4(shadowFrustaPlanes[12],shadowFrustaPlanes[13],shadowFrustaPlanes[14],shadowFrustaPlanes[15]);
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = 1.f-tr;
  status &= uint(dot(vec4(minCorner + tr*size,1.f),plane)>0.f);

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
      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level]+ node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
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


      if(gl_LocalInvocationIndex == 0u){
        if(unpackUint2x32(trA)[0] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+0],~unpackUint2x32(trA)[0]);
        if(unpackUint2x32(trA)[1] != 0u)
          atomicAnd(nodePool[nodeLevelOffsetInUints[level]+node*uintsPerWarp+1],~unpackUint2x32(trA)[1]);
      }
    }

    if(intersection[level] == 0){
      node >>= warpBits;
      level--;
    }else{
      uint selectedBit = unpackUint2x32(intersection[level])[0]!=0?findLSB(unpackUint2x32(intersection[level])[0]):findLSB(unpackUint2x32(intersection[level])[1])+32u;
      node <<= warpBits   ;
      node  += selectedBit;

      intersection[level] ^= 1u << selectedBit;
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

    traverse();

  }
}

).";
