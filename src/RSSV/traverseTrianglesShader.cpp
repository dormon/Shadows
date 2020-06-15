#include <RSSV/traverseTrianglesShader.h>

std::string const rssv::traverseTrianglesFWD = R".(
void traverseTriangleJOB();

#define PLANES_PER_SF    (4u + MORE_PLANES*3u + EXACT_TRIANGLE_AABB*3u)
#define FLOATS_PER_PLANE 4u
#define FLOATS_PER_SF    PLANES_PER_SF * FLOATS_PER_PLANE
const uint floatsPerSF = FLOATS_PER_SF;

#if !defined(SHARED_MEMORY_SIZE) || (SHARED_MEMORY_SIZE) < FLOATS_PER_SF
#undef SHARED_MEMORY_SIZE
#define SHARED_MEMORY_SIZE FLOATS_PER_SF
#endif
).";

std::string const rssv::traverseTriangles = R".(
const uint alignedNofSF        = (uint(NOF_TRIANGLES /       SF_ALIGNMENT) + uint((NOF_TRIANGLES %       SF_ALIGNMENT) != 0u)) *       SF_ALIGNMENT;


#define tri_abPlaneO       (0*4)
#define tri_bcPlaneO       (1*4)
#define tri_caPlaneO       (2*4)
#define tri_trianglePlaneO (3*4)

#if MORE_PLANES == 1
  #define tri_addPlane0O     (4*4)
  #define tri_addPlane1O     (5*4)
  #define tri_addPlane2O     (6*4)
#endif

#if MORE_PLANES == 1 && EXACT_TRIANGLE_AABB == 1
  #define tri_AO             (7*4)
  #define tri_BO             (8*4)
  #define tri_CO             (9*4)
#endif

#if MORE_PLANES == 0 && EXACT_TRIANGLE_AABB == 1
  #define tri_AO             (4*4)
  #define tri_BO             (5*4)
  #define tri_CO             (6*4)
#endif


#define tri_abPlane       getShared4f(tri_abPlaneO      )
#define tri_bcPlane       getShared4f(tri_bcPlaneO      )
#define tri_caPlane       getShared4f(tri_caPlaneO      )
#define tri_trianglePlane getShared4f(tri_trianglePlaneO)

#if MORE_PLANES == 1
#define tri_addPlane0    getShared4f(tri_addPlane0O)
#define tri_addPlane1    getShared4f(tri_addPlane1O)
#define tri_addPlane2    getShared4f(tri_addPlane2O)
#endif

#if EXACT_TRIANGLE_AABB == 1
#define tri_A            getShared4f(tri_AO)
#define tri_B            getShared4f(tri_BO)
#define tri_C            getShared4f(tri_CO)
#endif

#define shadowFrustaPlanes sharedMemory

void loadTriangle(uint job){
  if(gl_LocalInvocationIndex < floatsPerSF){
#if SF_INTERLEAVE == 1
    toShared1f(gl_LocalInvocationIndex,shadowFrusta[alignedNofSF*gl_LocalInvocationIndex + job]);
#else
    toShared1f(gl_LocalInvocationIndex,shadowFrusta[job*floatsPerSF+gl_LocalInvocationIndex]);
#endif
  }
#if WARP == 32
  memoryBarrierShared();
#endif
}

#if EXACT_TRIANGLE_AABB == 1
uint computeAABBTriangleIntersetionEXACT(in vec3 minCorner,in vec3 maxCorner){
  if(doesSubFrustumDiagonalIntersectTriangle(minCorner,maxCorner,tri_A,tri_B,tri_C,tri_trianglePlane))return INTERSECTS;
  if(doesLineInterectSubFrustum(tri_A,tri_B,minCorner,maxCorner))return INTERSECTS;
  if(doesLineInterectSubFrustum(tri_B,tri_C,minCorner,maxCorner))return INTERSECTS;
  if(doesLineInterectSubFrustum(tri_C,tri_A,minCorner,maxCorner))return INTERSECTS;
  return TRIVIAL_REJECT;
}
#endif

uint computeAABBTriangleIntersetion(in vec3 minCorner,in vec3 size){
  uint status = TRIVIAL_ACCEPT;
  vec4 plane;
  vec3 tr;
  //if(minCorner.x != 1337)return TRIVIAL_REJECT;

  //plane = tri_trianglePlane;
  //tr    = trivialRejectCorner3D(plane.xyz);
  //if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
  //  return TRIVIAL_REJECT;
  //tr = vec3(1.f)-tr;
  //if(dot(plane,vec4(minCorner + tr*size,1.f))>0.f)
  //  return TRIVIAL_REJECT;



  plane = tri_abPlane;
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = tri_bcPlane;
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = tri_caPlane;
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1),plane)>0.f);

  plane = tri_trianglePlane;
  tr    = trivialRejectCorner3D(plane.xyz);
  if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
    return TRIVIAL_REJECT;
  tr = vec3(1.f)-tr;
  if(dot(plane,vec4(minCorner + tr*size,1.f))>0.f)
    return TRIVIAL_REJECT;
  status &= 2u+uint(dot(vec4(minCorner + tr*size,1.f),plane)>0.f);


#if MORE_PLANES == 1
  if(status == INTERSECTS){
    plane = tri_addPlane0;
    tr    = trivialRejectCorner3D(plane.xyz);
    if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
      return TRIVIAL_REJECT;

    plane = tri_addPlane1;
    tr    = trivialRejectCorner3D(plane.xyz);
    if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
      return TRIVIAL_REJECT;

    plane = tri_addPlane2;
    tr    = trivialRejectCorner3D(plane.xyz);
    if(dot(plane,vec4(minCorner + tr*size,1.f))<0.f)
      return TRIVIAL_REJECT;
  }
#endif



  return status;
}

void computeBridgeTriangleIntersection(in vec3 minCorner,in vec3 aabbSize,int level,uint node){
#if COMPUTE_TRIANGLE_BRIDGES == 1 && EXACT_TRIANGLE_AABB == 1
  //TODO computation when we dont have tri_A,tri_B,tri_C if EXACT == false
  vec4 bridgeStart;
  vec4 bridgeEnd  ;


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

  int mult = 2*doesLineIntersectTriangle(bridgeStart,bridgeEnd,tri_A,tri_B,tri_C,clipLightPosition);
  if(mult!=0)atomicAdd(bridges[nodeLevelOffset[level] + node*WARP + gl_LocalInvocationIndex],mult);
  //if(level == 0 && gl_LocalInvocationIndex == 34){
  //  mat4 nodeTran = projView;
  //  //nodeTran[0][0] = uintBitsToFloat(1060262557u);
  //  //nodeTran[0][1] = uintBitsToFloat(3200271021u);
  //  //nodeTran[0][2] = uintBitsToFloat(3206316816u);
  //  //nodeTran[0][3] = uintBitsToFloat(3206316816u);
  //  //nodeTran[1][0] = uintBitsToFloat(         0u);
  //  //nodeTran[1][1] = uintBitsToFloat(1062871986u);
  //  //nodeTran[1][2] = uintBitsToFloat(3204840296u);
  //  //nodeTran[1][3] = uintBitsToFloat(3204840296u);
  //  //nodeTran[2][0] = uintBitsToFloat(3208097063u);
  //  //nodeTran[2][1] = uintBitsToFloat(3199903766u);
  //  //nodeTran[2][2] = uintBitsToFloat(3206017847u);
  //  //nodeTran[2][3] = uintBitsToFloat(3206017847u);
  //  //nodeTran[3][0] = uintBitsToFloat(3221392974u);
  //  //nodeTran[3][1] = uintBitsToFloat(1063916002u);
  //  //nodeTran[3][2] = uintBitsToFloat(1091679437u);
  //  //nodeTran[3][3] = uintBitsToFloat(1091889152u);

  //  
  //  
  //  //vec4 L = nodeTran*vec4( 0,5, 0,1);
  //  vec4 A = nodeTran*vec4( 1,2, 1,1);
  //  vec4 B = nodeTran*vec4( 1,2,-1,1);
  //  vec4 C = nodeTran*vec4(-1,2, 1,1);
  //  //vec3 mmin;
  //  //vec3 mmax;
  //  //mmin.x = uintBitsToFloat(3207585792u);
  //  //mmax.x = uintBitsToFloat(3120562176u);
  //  //mmin.y = uintBitsToFloat( 973078528u);
  //  //mmax.y = uintBitsToFloat(1050853376u);
  //  //mmin.z = uintBitsToFloat(1064877835u);
  //  //mmax.z = uintBitsToFloat(1065008125u);
  //  //vec4 newEnd = vec4((mmin+mmax)/2.f,1);
  //  int mult = abs(doesLineIntersectTriangle(clipLightPosition,bridgeEnd,A,B,C,clipLightPosition));

  //  if(mult!=0)atomicAdd(bridges[nodeLevelOffset[level] + node*WARP + gl_LocalInvocationIndex],mult);
  //}else{
  //  int mult = 2*doesLineIntersectTriangle(bridgeStart,bridgeEnd,tri_A,tri_B,tri_C,clipLightPosition);
  //  if(mult!=0)atomicAdd(bridges[nodeLevelOffset[level] + node*WARP + gl_LocalInvocationIndex],mult);
  //}
#endif
}

void debug_storeTriangleTraverseStatLastLevel(in uint job,in uint node,in int level){
#if STORE_TRIANGLE_TRAVERSE_STAT == 1
  if(gl_LocalInvocationIndex==0){
    uint w = atomicAdd(debug[0],1);
    debug[1+w*4+0] = job;
    debug[1+w*4+1] = node;
    debug[1+w*4+2] = uint(level);
    debug[1+w*4+3] = 0xff;
  }
#endif
}

void debug_storeTriangleTraverseStat(in uint job,in uint node,in int level,uint status){
#if STORE_TRIANGLE_TRAVERSE_STAT == 1
  uint w = atomicAdd(debug[0],1);
  debug[1+w*4+0] = job;
  debug[1+w*4+1] = node*WARP + gl_LocalInvocationIndex;
  debug[1+w*4+2] = uint(level);
  debug[1+w*4+3] = status;
#endif
}

void lastLevelTriangles(uint node){
#if COMPUTE_LAST_LEVEL_TRIANGLES == 1
  uvec2 sampleCoord = (demorton(node).xy<<uvec2(tileBitsX,tileBitsY)) + uvec2(gl_LocalInvocationIndex&tileMaskX,gl_LocalInvocationIndex>>tileBitsX);

  vec4 bridgeEnd;
  bridgeEnd.z = texelFetch(depthTexture,ivec2(sampleCoord)).x*2-1;
  bridgeEnd.xy = -1+2*((vec2(sampleCoord) + vec2(0.5)) / vec2(WINDOW_X,WINDOW_Y));
  bridgeEnd.w = 1.f;

  vec4 bridgeStart = vec4(getAABBCenter(nofLevels-1,node),1.f);

  int mult = 2*doesLineIntersectTriangle(bridgeStart,bridgeEnd,tri_A,tri_B,tri_C,clipLightPosition);

  if(mult!=0)imageAtomicAdd(stencil,ivec2(sampleCoord),mult);
#endif
}


void traverseTriangle(uint job){
  int level = 0;

  uint64_t currentIntersection;

  uint node = 0;
  while(level >= 0){
    if(level == int(nofLevels)){

      debug_storeTriangleTraverseStatLastLevel(job,node,level);

      lastLevelTriangles(node);

      node >>= warpBits;
      level--;
    }else{
      uint status = uint(nodePool[nodeLevelOffsetInUints[level] + node*uintsPerWarp + uint(gl_LocalInvocationIndex>31u)]&uint(1u<<(gl_LocalInvocationIndex&0x1fu)));
      if(status != 0u){

        if(level >  int(nofLevels))return;

        vec3 minCorner;
        vec3 maxCorner;
        getAABB(minCorner,maxCorner,level,(node<<warpBits)+gl_LocalInvocationIndex);

#if EXACT_TRIANGLE_AABB == 1
  #if EXACT_TRIANGLE_AABB_LEVEL < 0
        status = computeAABBTriangleIntersetionEXACT(minCorner,maxCorner);
        maxCorner -= minCorner;
  #else
        if(level <= EXACT_TRIANGLE_AABB_LEVEL){
          status = computeAABBTriangleIntersetionEXACT(minCorner,maxCorner);
          maxCorner -= minCorner;
        }else{
          maxCorner -= minCorner;
          status = computeAABBTriangleIntersetion(minCorner,maxCorner);
        }
  #endif
#else
        maxCorner -= minCorner;
        status = computeAABBTriangleIntersetion(minCorner,maxCorner);
#endif
        if(status != INTERSECTS)status = TRIVIAL_REJECT;

        computeBridgeTriangleIntersection(minCorner,maxCorner,level,node);

      }

      debug_storeTriangleTraverseStat(job,node,level,status);

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

void traverseTriangleJOB(){
#if PERFORM_TRAVERSE_TRIANGLES == 1
  uint job = 0u;
  //triangle loop
  for(;;){
  
    if(gl_LocalInvocationIndex==0)
      job = atomicAdd(triangleJobCounter,1);

    job = readFirstInvocationARB(job);
    if(job >= NOF_TRIANGLES)break;

    loadTriangle(job);
    traverseTriangle(job);
  }
#endif
}
).";

