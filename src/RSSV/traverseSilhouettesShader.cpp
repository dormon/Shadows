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

//#pragma debug(on)

layout(local_size_x=WARP)in;

#if MERGED_BUFFERS == 1
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

uniform mat4 invTran;
uniform mat4 projView;


shared int  edgeMult;

shared vec4 edgePlane;
shared vec4 aPlane   ;
shared vec4 bPlane   ;
shared vec4 abPlane  ;

shared vec4 edgeAClipSpace;
shared vec4 edgeBClipSpace;
shared vec4 lightClipSpace;

#line 52

// Silhouette VS AABB and BRIDGE
// Input:
// edge (A,B) worldspace
// light (L) worldspace / clipspace
// AABB: clip space minCorner + size
// Bridge: clip space bridgeStart(M) bridgeEnd(N)
//
// col(AABB,sil):
//   Sil: aPlane (AL) bPlane (BL) abPlane (AB) edgePlane (ABL)
//   AABB: clip
//   for collision we need aPlane bPlane abPlane edgePlane
//   edgePlane = proj*plane(ABL)        noclip(ABL)
//   aPlane = proj*plane(A,L,nor(ABL))  noclip(ABL)
//   bPlane = proj*plane(B,L,nor(ABL))  noclip(ABL)
//   abPlane = proj*plane(A,B,nor(ABL)) noclip(ABL)
//
// col(bridge,sil):
//   bridge: M N clip
//   Sil: edgePlane (ABL) samplePlane (MNL) trianglePlane (ABMN)
//   edgePlane = proj*plane(ABL) - same as for AABB
//   samplePlane = skala(M,N,L) - new
//   trianglePlane = 
//   
// //separate issue
// col(AABB,tri):
//   shadowFrustaPlanes needed
//
// col(bridge,tri):
//   trianglePlane (ABMN) (BCMN) (CAMN)
//
//


#if STORE_EDGE_PLANES == 1
layout(std430,binding = 7)buffer Debug{uint debug[];};
#endif

//shared vec3 edgeAE;
//shared vec3 edgeBE;

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

    ////debug
    //edgeAE = edgeA;
    //edgeBE = edgeB;

    //edgeA = vec3(-1,2,1);
    //edgeB = vec3(1,2,-1);

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
    //edgeAClipSpace = projView*vec4(edgeA,1.f);
    //edgeBClipSpace = projView*vec4(edgeB,1.f);
    //lightClipSpace = projView*lightPosition  ;

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

//int computeBridge(in vec3 bridgeStart,in vec3 bridgeEnd){
//  // m n c 
//  // - - 0
//  // 0 - 1
//  // + - 1
//  // - 0 1
//  // 0 0 0
//  // + 0 0
//  // - + 1
//  // 0 + 0
//  // + + 0
//  //
//  // m<0 && n>=0 || m>=0 && n<0
//  // m<0 xor n<0
//
//  int result = edgeMult;
//  float ss = dot(edgePlane,vec4(bridgeStart,1));
//  float es = dot(edgePlane,vec4(bridgeEnd  ,1));
//  if((ss<0)==(es<0))return 0;
//  result *= 1-2*int(ss<0.f);
//
//  vec4 samplePlane    = getClipPlaneSkala(vec4(bridgeStart,1),vec4(bridgeEnd,1),lightClipSpace);
//  ss = dot(samplePlane,edgeAClipSpace);
//  es = dot(samplePlane,edgeBClipSpace);
//  ss*=es;
//  if(ss>0.f)return 0;
//  result *= 1+int(ss<0.f);
//
//  vec4 trianglePlane  = getClipPlaneSkala(vec4(bridgeStart,1),vec4(bridgeEnd,1),vec4(bridgeStart,1) + (edgeBClipSpace-edgeAClipSpace));
//  trianglePlane *= sign(dot(trianglePlane,lightClipSpace));
//  if(dot(trianglePlane,edgeAClipSpace)<=0)return 0;
//
//  return result;
//
//}

//vec4 plane(vec3 a,vec3 b,vec3 c){
//  vec3 n = normalize(cross(b-a,c-a));
//  return vec4(n,-dot(n,a));
//}
//
//int computeBridgeEuclid(in vec4 bridgeStart,in vec4 bridgeEnd){
//  vec4 bs = inverse(projView)*bridgeStart;
//  vec4 be = inverse(projView)*bridgeEnd  ;
//  bs/=bs.w;
//  be/=be.w;
//
//  //edgeAE = vec3(-1,2,1);
//  //edgeBE = vec3( 1,2,1);
//  //vec4 edgePlaneE = plane(edgeAE,edgeBE,lightPosition.xyz);
//  vec4 edgePlaneE = plane(vec3(-1,2,1),vec3( 1,2,1),vec3(0,5,0));
//
//  int result = edgeMult;
//  float ss = dot(edgePlaneE,bs);
//  float es = dot(edgePlaneE,be);
//  //if((ss<0)==(es<0))return 0;
//  //result *= 1-2*int(ss<0.f);
//  ////return result;
//  if((ss<0))return  1;
//  if((ss>0))return -1;
//  return 0;//abs(result);
//
//  vec4 samplePlane = plane(bs.xyz,be.xyz,lightPosition.xyz);
//  ss = dot(samplePlane,vec4(edgeAE,1));
//  es = dot(samplePlane,vec4(edgeBE,1));
//  ss*=es;
//  if(ss>0.f)return 0;
//  result *= 1+int(ss<0.f);
//
//
//  vec4 trianglePlane  = plane(bs.xyz,be.xyz,bs.xyz + (edgeBE-edgeAE));
//  trianglePlane *= sign(dot(trianglePlane,lightPosition));
//  if(dot(trianglePlane,vec4(edgeAE,1))<=0)return 0;
//
//  return result;
//}

int computeBridge(in vec4 bridgeStart,in vec4 bridgeEnd){
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
  //
  //
  
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

void lastLevel(uint node){
  uvec2 sampleCoord = (demorton(node).xy<<uvec2(tileBitsX,tileBitsY)) + uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);

  vec4 bridgeEnd;
  bridgeEnd.z = texelFetch(depthTexture,ivec2(sampleCoord)).x*2-1;
  bridgeEnd.xy = -1+2*((vec2(sampleCoord) + vec2(0.5)) / vec2(WINDOW_X,WINDOW_Y));
  bridgeEnd.w = 1.f;

  vec4 bridgeStart = vec4(getAABBCenter(nofLevels-1,node),1.f);

  int mult = computeBridge(bridgeStart,bridgeEnd);

  if(mult!=0)imageAtomicAdd(stencil,ivec2(sampleCoord),mult);
}

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
shared vec3     localBridgeEnd   [nofLevels][WARP];
#endif
#endif


//int computeBridgeMultiplicity(){
//#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
//  return computeBridge(bridgeStart,bridgeEnd[level][gl_LocalInvocationIndex]);
//#else
//  vec3 bridgeStart = loadAABBCenter(level-1,node>>warpBits);
//  return computeBridge(bridgeStart,minCorner + aabbSize/2.f)+1;
//#endif
//}


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

#if COMPUTE_LAST_LEVEL_SILHOUETTES == 1
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
        getAABB(minCorner,aabbSize,level,(node<<warpBits)+gl_LocalInvocationIndex);
        aabbSize -= minCorner;


#if COMPUTE_BRIDGES == 1
        vec4 bridgeStart;
        vec4 bridgeEnd  ;
        int  mult       ;

  
        bridgeEnd = vec4(minCorner + aabbSize*.5f,1.f);

  #if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
        localBridgeEnd[level][gl_LocalInvocationIndex] = vec3(bridgeEnd);
  #endif
  
        if(level == 0){
          bridgeStart = proj*view*lightPosition;
        }else{
  #if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
          bridgeStart = vec4(localBridgeEnd[level-1][node&warpMask],1);
  #else
        bridgeStart = vec4(getAABBCenter(level-1,node),1);
  #endif
        }

        mult = computeBridge(bridgeStart,bridgeEnd);

        //mult = computeBridgeEuclid(bridgeStart,bridgeEnd);
        if(mult!=0)atomicAdd(bridges[nodeLevelOffsetInUints[level] + node*WARP + gl_LocalInvocationIndex],mult);
 
  
//  #if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
//          bridgeEnd[level][gl_LocalInvocationIndex] = minCorner + aabbSize * 0.5f;
//  
//          vec3 bridgeStart;
//          if(level == 0)
//            bridgeStart = lightPosition.xyz;
//          else
//            bridgeStart = bridgeEnd[level-1][node&warpMask];
//  #endif
//  
//          if(level > 0){
//  #if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
//            int mult = computeBridge(bridgeStart,bridgeEnd[level][gl_LocalInvocationIndex]);
//  #else
//
//            vec3 bridgeStart = loadAABBCenter(level-1,node>>warpBits);
//            int mult = computeBridge(bridgeStart,minCorner + aabbSize/2.f)+1;
//  #endif
//            if(mult!=0)atomicAdd(bridges[nodeLevelOffsetInUints[level] + node*WARP + gl_LocalInvocationIndex],mult);
//          }
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

uniform int selectedEdge = -1;

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

    if(selectedEdge>=0 && job != uint(selectedEdge))continue;
    //job=3;
    loadSilhouette(job);

    traverse();
    //break;

  }
}
).";
