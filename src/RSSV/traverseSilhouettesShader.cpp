#include <RSSV/traverseSilhouettesShader.h>

std::string const rssv::traverseSilhouettesShader = R".(

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

layout(std430,binding=5)readonly buffer Triangles{float triangles[];};

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

#define tri_trianglePlane sharedVec4[0]
#define tri_abPlane       sharedVec4[1]
#define tri_bcPlane       sharedVec4[2]
#define tri_caPlane       sharedVec4[3]

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
  if(mult!=0)atomicAdd(bridges[nodeLevelOffsetInUints[level] + node*WARP + gl_LocalInvocationIndex],mult);
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
#if 0
  vec3 tr;
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

#if 0
void loadTriangle(uint job){
  if(gl_LocalInvocationIndex == 0){
    vec3 A;
    vec3 B;
    vec3 C;
    A[0] = triangles[job*9+0*3+0];
    A[1] = triangles[job*9+0*3+1];
    A[2] = triangles[job*9+0*3+2];
    B[0] = triangles[job*9+1*3+0];
    B[1] = triangles[job*9+1*3+1];
    B[2] = triangles[job*9+1*3+2];
    C[0] = triangles[job*9+2*3+0];
    C[1] = triangles[job*9+2*3+1];
    C[2] = triangles[job*9+2*3+2];
    vec3 n = normalize(cross(B-A,C-B));
    tri_trianglePlane = invTran*vec4(n,-dot(n,A));
    if(dot(tri_trianglePlane,clipLightPosition)<0){
      tri_trianglePlane *= -1;
      n *= -1;
    }
    vec3 n2;

    n2 = normalize(cross(n,B-A));
    tri_abPlane = invTran*vec4(n2,-dot(n2,A));
    n2 = normalize(cross(n,C-B));
    tri_bcPlane = invTran*vec4(n2,-dot(n2,B));
    n2 = normalize(cross(n,A-C));
    tri_caPlane = invTran*vec4(n2,-dot(n2,C));
  }
}

void computeAABBTriangleIntersection(out uint status,in vec3 minCorner,in vec3 aabbSize){
  status = TRIVIAL_REJECT;
  vec3 tr;
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
}

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

        computeAABBTriangleIntersection(status,minCorner,aabbSize);
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

#if 0
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
