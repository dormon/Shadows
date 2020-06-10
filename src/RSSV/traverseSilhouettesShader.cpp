#include <RSSV/traverseSilhouettesShader.h>

std::string const rssv::traverseSilhouettesFWD = R".(
void traverseSilhouetteJOB();
#if !defined(SHARED_MEMORY_SIZE) || (SHARED_MEMORY_SIZE) < 7*4+1
#undef SHARED_MEMORY_SIZE
#define SHARED_MEMORY_SIZE 7*4+1
#endif
).";

std::string const extern rssv::traverseSilhouettes = R".(

#define edgePlaneO      (0*4)
#define aPlaneO         (1*4)
#define bPlaneO         (2*4)
#define abPlaneO        (3*4)
#define edgeAClipSpaceO (4*4)
#define edgeBClipSpaceO (5*4)
#define lightClipSpaceO (6*4)
#define edgeMultO       (7*4)

#define edgePlane      getShared4f(edgePlaneO     )
#define aPlane         getShared4f(aPlaneO        )
#define bPlane         getShared4f(bPlaneO        )
#define abPlane        getShared4f(abPlaneO       )
#define edgeAClipSpace getShared4f(edgeAClipSpaceO)
#define edgeBClipSpace getShared4f(edgeBClipSpaceO)
#define lightClipSpace getShared4f(lightClipSpaceO)
#define edgeMult       getShared1i(edgeMultO      )

void loadSilhouette(uint job){
  if(gl_LocalInvocationIndex == 0){
    uint res  = multBuffer[job];
    uint edge = res & 0x1fffffffu;
    int  mult = int(res) >> 29;

    vec3 edgeA;
    vec3 edgeB;
    loadEdge(edgeA,edgeB,edge);

    vec3 n = normalize(cross(edgeB-edgeA,lightPosition.xyz-edgeA));
    toShared(edgePlaneO,invTran*vec4(n  ,-dot(n  ,edgeA)));

    vec3 an = normalize(cross(n,edgeA-lightPosition.xyz));
    toShared(aPlaneO   ,invTran*vec4(an ,-dot(an ,edgeA)));

    vec3 bn = normalize(cross(edgeB-lightPosition.xyz,n));
    toShared(bPlaneO   ,invTran*vec4(bn ,-dot(bn ,edgeB)));

    vec3 abn = normalize(cross(edgeB-edgeA,n));
    toShared(abPlaneO  ,invTran*vec4(abn,-dot(abn,edgeA)));

#if COMPUTE_BRIDGES == 1
    toShared(edgeAClipSpaceO,projView*vec4(edgeA,1.f));
    toShared(edgeBClipSpaceO,projView*vec4(edgeB,1.f));
    toShared(lightClipSpaceO,clipLightPosition       );
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
    

    toShared(edgeMultO,mult);
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


#if WARP == 64

#if COMPUTE_BRIDGES == 1
#if STORE_BRIDGES_IN_LOCAL_MEMORY == 1
shared vec3     localBridgeEnd   [nofLevels][WARP];
#endif
#endif

void debug_storeSilhouetteTraverseStatLastLevel(in uint job,in uint node,in int level){
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

void debug_storeSilhouetteTraverseStat(in uint job,in uint node,in int level,uint status){
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

void traverseSilhouette(uint job){
  int level = 0;

  uint64_t currentIntersection;

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){

      //debug_storeSilhouetteTraverseStatLastLevel(job,node,level);

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

      //debug_storeSilhouetteTraverseStat(job,node,level,status);

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
void traverseSilhouetteJOB(){
#if PERFORM_TRAVERSE_SILHOUETTES == 1
  uint job = 0u;
  //silhouette loop
  for(;;){
    if(gl_LocalInvocationIndex==0)
      job = atomicAdd(silhouetteJobCounter,1);

    job = readFirstInvocationARB(job);
    if(job >= nofSilhouettes)break;

    //if(selectedEdge>=0 && job != uint(selectedEdge))continue;

    loadSilhouette(job);
    traverseSilhouette(job);

  }
#endif
}

).";

