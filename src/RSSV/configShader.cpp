#include <RSSV/configShader.h>

std::string const rssv::configShader = R".(
#line 8000
#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef WINDOW_X
#define WINDOW_X 512
#endif//WINDOW_X

#ifndef WINDOW_Y
#define WINDOW_Y 512
#endif//WINDOW_Y

#ifndef TILE_X
#define TILE_X 8
#endif//TILE_X

#ifndef TILE_Y
#define TILE_Y 8
#endif//TILE_Y

#ifndef MIN_Z_BITS
#define MIN_Z_BITS 9
#endif//MIN_Z_BITS

#ifndef NEAR
#define NEAR 0.01f
#endif//NEAR

#ifndef FAR
#define FAR 1000.f
#endif//FAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY


#define DIV_ROUND_UP(x,y) uint(uint(uint(x)/uint(y)) + uint((uint(x) % uint(y)) != 0u))
#define BITS_REQUIRED(x) uint(ceil(log2(float(x))))
#line 31
const uint tileBitsX       = BITS_REQUIRED(TILE_X);
const uint tileBitsY       = BITS_REQUIRED(TILE_Y);
const uint tileMaskX       = uint(TILE_X-1u);
const uint tileMaskY       = uint(TILE_Y-1u);
const uint warpBits        = BITS_REQUIRED(WARP);
const uint clustersX       = DIV_ROUND_UP(WINDOW_X,TILE_X);
const uint clustersY       = DIV_ROUND_UP(WINDOW_Y,TILE_Y);
const uint xBits           = BITS_REQUIRED(clustersX);
const uint yBits           = BITS_REQUIRED(clustersY);
const uint zBits           = MIN_Z_BITS>0?MIN_Z_BITS:max(max(xBits,yBits),MIN_Z_BITS);
const uint clustersZ       = 1u << zBits;
const uint allBits         = xBits + yBits + zBits;
const uint nofLevels       = DIV_ROUND_UP(allBits,warpBits);
const uint uintsPerWarp    = uint(WARP/32u);
const uint noAxis          = 3u;
#line 47
const uint bitLength[3] = {
  min(min(xBits,yBits),zBits),
  max(max(min(xBits,yBits),min(xBits,zBits)),min(yBits,zBits)),
  max(max(xBits,yBits),zBits),
};
#line 53
const uint bitTogether[3] = {
  bitLength[0]                   ,
  uint(bitLength[1]-bitLength[0]),
  uint(bitLength[2]-bitLength[1]),
};

const uint longestAxis  = 
  bitLength[2]==zBits?2u:
  bitLength[2]==yBits?1u:
  0u;

const uint shortestAxis = 
  bitLength[0]==xBits?0u:
  bitLength[0]==yBits?1u:
  2u;
const uint middleAxis   = 
  shortestAxis==0u?(longestAxis==1u?2u:1u):
  (shortestAxis==1u?(longestAxis==0u?2u:0u):
  (longestAxis==0u?1u:0u));

const uint twoLongest[] = {
  0==shortestAxis?1u:0u,
  2==shortestAxis?1u:2u,
};

#define CPTSV_QUANTIZATION(z)   clamp(uint(log(-z/NEAR) / log(1.f+2.f*tan(FOVY/2.f)/clustersY)),0u,clustersZ-1u)
#define CPTSV_DEQUANTIZATION(i) (-NEAR * exp((i)*log(1.f + 2.f*tan(FOVY/2.f)/clustersY)))

#if SCALED_QUANTIZATION == 1
  const float maxZOfHierarchy = -CPTSV_DEQUANTIZATION(clustersZ);
  
  #ifdef FAR_IS_INFINITE
    #define SCALED_QUANTIZATION_FAR 10000.f
  #else
    #define SCALED_QUANTIZATION_FAR FAR
  #endif

  const float z2ScaledZFactor = (maxZOfHierarchy-NEAR) / (SCALED_QUANTIZATION_FAR-NEAR);
  const float scaledZ2ZFactor = (SCALED_QUANTIZATION_FAR-NEAR) / (maxZOfHierarchy-NEAR);

  #define ORIGINAL_Z_TO_SCALED_Z(z) (((z)+NEAR)*z2ScaledZFactor-NEAR)
  #define SCALED_Z_TO_ORIGINAL_Z(z) (((z)+NEAR)*scaledZ2ZFactor-NEAR)

  #define QUANTIZE_Z(z)   CPTSV_QUANTIZATION(ORIGINAL_Z_TO_SCALED_Z(z))  
  #define CLUSTER_TO_Z(i) SCALED_Z_TO_ORIGINAL_Z(CPTSV_DEQUANTIZATION(i))
#else
  #define QUANTIZE_Z(z)   CPTSV_QUANTIZATION(z)
  #define CLUSTER_TO_Z(i) CPTSV_DEQUANTIZATION(i)
#endif

// | 2n/(R-L)  0          (R+L)/(R-L)  0          |   |x|
// | 0         2n/(T-B)   (T+B)/(T-B)  0          | * |y|
// | 0         0         -(f+n)/(f-n)  -2fn/(f-n) |   |z|
// | 0         0         -1            0          |   |1|
//
// ndcdepth<-1,1> = (-(f+n)/(f-n)*z  -2fn/(f-n)*1)/(-z)
// d = (-(f+n)/(f-n)*z  -2fn/(f-n)*1)/(-z)
// d = (f+n)/(f-n) + 2fn/(f-n)/z
// d-(f+n)/(f-n) = 2fn/(f-n)/z
// z = 2fn/(f-n) / (d-(f+n)/(f-n))
// z = 2fn/(f-n) / ((d*(f-n)-(f+n))/(f-n))
// z = 2fn / ((d*(f-n)-(f+n)))
// z = 2fn / ((d*(f-n)-f-n)))
//

#ifdef FAR_IS_INFINITE
  #define DEPTH_TO_Z(d) (2.f*NEAR    /((d) - 1.f))
  #define Z_TO_DEPTH(z) ((2.f*NEAR)/(z)+1.f)
#else
  #define DEPTH_TO_Z(d) (2.f*NEAR*FAR/((d)*(FAR-NEAR)-FAR-NEAR))
  #define Z_TO_DEPTH(z) (((2.f*NEAR*FAR/(z))+FAR+NEAR)/(FAR-NEAR))
#endif


#define GET_MASK(x) uint((x)==32u?0xffffffffu:uint(uint(1u<<(x))-1u))

const uvec3 bitPosition = uvec3(
  ((0x49249249u<<0u) & GET_MASK(bitTogether[0]*3u))|
  (uint(shortestAxis != 0) * (((0x55555555u<<0u                     ) & GET_MASK(bitTogether[1]*2u)) << (bitTogether[0]*3u)))|
  ((GET_MASK(bitTogether[2]) << (bitTogether[0]*3u + bitTogether[1]*2u))*uint(longestAxis == 0u)),

  ((0x49249249u<<1u) & GET_MASK(bitTogether[0]*3u))|
  (uint(shortestAxis != 1) * (((0x55555555u<<uint(shortestAxis != 0)) & GET_MASK(bitTogether[1]*2u)) << (bitTogether[0]*3u)))|
  ((GET_MASK(bitTogether[2]) << (bitTogether[0]*3u + bitTogether[1]*2u))*uint(longestAxis == 1u)),

  ((0x49249249u<<2u) & GET_MASK(bitTogether[0]*3u))|
  (uint(shortestAxis != 2) * (((0x55555555u<<1u                     ) & GET_MASK(bitTogether[1]*2u)) << (bitTogether[0]*3u)))|
  ((GET_MASK(bitTogether[2]) << (bitTogether[0]*3u + bitTogether[1]*2u))*uint(longestAxis == 2u))
);

#line 172
const uvec3 levelTileBits[] = {
  bitCount(bitPosition&((1u<<(warpBits*uint(max(int(nofLevels)-1,0))))-1u)),
  bitCount(bitPosition&((1u<<(warpBits*uint(max(int(nofLevels)-2,0))))-1u)),
  bitCount(bitPosition&((1u<<(warpBits*uint(max(int(nofLevels)-3,0))))-1u)),
  bitCount(bitPosition&((1u<<(warpBits*uint(max(int(nofLevels)-4,0))))-1u)),
  bitCount(bitPosition&((1u<<(warpBits*uint(max(int(nofLevels)-5,0))))-1u)),
  bitCount(bitPosition&((1u<<(warpBits*uint(max(int(nofLevels)-6,0))))-1u)),
};
#line 8000
const uvec3 levelTileSize[] = {                                            
  uvec3(1u)<<levelTileBits[0],
  uvec3(1u)<<levelTileBits[1],
  uvec3(1u)<<levelTileBits[2],
  uvec3(1u)<<levelTileBits[3],
  uvec3(1u)<<levelTileBits[4],
  uvec3(1u)<<levelTileBits[5],
};

const uvec3 levelTileSizeInPixels[] = {
  levelTileSize[0] << uvec3(tileBitsX,tileBitsY,0u),
  levelTileSize[1] << uvec3(tileBitsX,tileBitsY,0u),
  levelTileSize[2] << uvec3(tileBitsX,tileBitsY,0u),
  levelTileSize[3] << uvec3(tileBitsX,tileBitsY,0u),
  levelTileSize[4] << uvec3(tileBitsX,tileBitsY,0u),
  levelTileSize[5] << uvec3(tileBitsX,tileBitsY,0u),
};

const vec3 levelTileSizeClipSpace[] = {
  vec3(2.f * vec2(levelTileSizeInPixels[0].xy) / vec2(WINDOW_X,WINDOW_Y),CLUSTER_TO_Z(levelTileSizeInPixels[0].z)),
  vec3(2.f * vec2(levelTileSizeInPixels[1].xy) / vec2(WINDOW_X,WINDOW_Y),CLUSTER_TO_Z(levelTileSizeInPixels[0].z)),
  vec3(2.f * vec2(levelTileSizeInPixels[2].xy) / vec2(WINDOW_X,WINDOW_Y),CLUSTER_TO_Z(levelTileSizeInPixels[0].z)),
  vec3(2.f * vec2(levelTileSizeInPixels[3].xy) / vec2(WINDOW_X,WINDOW_Y),CLUSTER_TO_Z(levelTileSizeInPixels[0].z)),
  vec3(2.f * vec2(levelTileSizeInPixels[4].xy) / vec2(WINDOW_X,WINDOW_Y),CLUSTER_TO_Z(levelTileSizeInPixels[0].z)),
  vec3(2.f * vec2(levelTileSizeInPixels[5].xy) / vec2(WINDOW_X,WINDOW_Y),CLUSTER_TO_Z(levelTileSizeInPixels[0].z)),
};

const uint warpMask        = uint(WARP - 1u);
const uint floatsPerAABB   = 6u;

const uint halfWarp        = WARP / 2u;
const uint halfWarpMask    = uint(halfWarp - 1u);

const uint nodesPerLevel[6] = {
  (1u << uint(max(int(allBits) - int((nofLevels-1u)*warpBits),0)))*uint(nofLevels>=1u),
  (1u << uint(max(int(allBits) - int((nofLevels-2u)*warpBits),0)))*uint(nofLevels>=2u),
  (1u << uint(max(int(allBits) - int((nofLevels-3u)*warpBits),0)))*uint(nofLevels>=3u),
  (1u << uint(max(int(allBits) - int((nofLevels-4u)*warpBits),0)))*uint(nofLevels>=4u),
  (1u << uint(max(int(allBits) - int((nofLevels-5u)*warpBits),0)))*uint(nofLevels>=5u),
  (1u << uint(max(int(allBits) - int((nofLevels-6u)*warpBits),0)))*uint(nofLevels>=6u),
};
#line 70

const uint nofNodes = 
  nodesPerLevel[0] + 
  nodesPerLevel[1] + 
  nodesPerLevel[2] + 
  nodesPerLevel[3] + 
  nodesPerLevel[4] + 
  nodesPerLevel[5] ;

const uint nodeLevelOffset[6] = {
  0,
  0 + nodesPerLevel[0],
  0 + nodesPerLevel[0] + nodesPerLevel[1],
  0 + nodesPerLevel[0] + nodesPerLevel[1] + nodesPerLevel[2],
  0 + nodesPerLevel[0] + nodesPerLevel[1] + nodesPerLevel[2] + nodesPerLevel[3],
  0 + nodesPerLevel[0] + nodesPerLevel[1] + nodesPerLevel[2] + nodesPerLevel[3] + nodesPerLevel[4],
};

const uint nodeLevelSizeInUints[6] = {
  DIV_ROUND_UP(nodesPerLevel[0],WARP) * uintsPerWarp,
  DIV_ROUND_UP(nodesPerLevel[1],WARP) * uintsPerWarp,
  DIV_ROUND_UP(nodesPerLevel[2],WARP) * uintsPerWarp,
  DIV_ROUND_UP(nodesPerLevel[3],WARP) * uintsPerWarp,
  DIV_ROUND_UP(nodesPerLevel[4],WARP) * uintsPerWarp,
  DIV_ROUND_UP(nodesPerLevel[5],WARP) * uintsPerWarp,
};

const uint nodeBufferSizeInUints = 
  nodeLevelSizeInUints[0] + 
  nodeLevelSizeInUints[1] + 
  nodeLevelSizeInUints[2] + 
  nodeLevelSizeInUints[3] + 
  nodeLevelSizeInUints[4] + 
  nodeLevelSizeInUints[5] ; 

const uint nodeLevelOffsetInUints[6] = {
  0,
  0 + nodeLevelSizeInUints[0],
  0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1],
  0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1] + nodeLevelSizeInUints[2],
  0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1] + nodeLevelSizeInUints[2] + nodeLevelSizeInUints[3],
  0 + nodeLevelSizeInUints[0] + nodeLevelSizeInUints[1] + nodeLevelSizeInUints[2] + nodeLevelSizeInUints[3] + nodeLevelSizeInUints[4],
};

const uint aabbLevelSizeInFloats[6] = {
  nodesPerLevel[0] * floatsPerAABB,
  nodesPerLevel[1] * floatsPerAABB,
  nodesPerLevel[2] * floatsPerAABB,
  nodesPerLevel[3] * floatsPerAABB,
  nodesPerLevel[4] * floatsPerAABB,
  nodesPerLevel[5] * floatsPerAABB,
};

#if MEMORY_OPTIM == 1
const uint aabbBufferSizeInFloats = clustersX * clustersY * floatsPerAABB * MEMORY_FACTOR;
#else
const uint aabbBufferSizeInFloats = 
  aabbLevelSizeInFloats[0] + 
  aabbLevelSizeInFloats[1] + 
  aabbLevelSizeInFloats[2] + 
  aabbLevelSizeInFloats[3] + 
  aabbLevelSizeInFloats[4] + 
  aabbLevelSizeInFloats[5] ; 
#endif

const uint aabbLevelOffsetInFloats[6] = {
  0,
  0 + aabbLevelSizeInFloats[0],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2] + aabbLevelSizeInFloats[3],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2] + aabbLevelSizeInFloats[3] + aabbLevelSizeInFloats[4],
};

const uint aabbPointerBufferSizeInUints = nofNodes + 1;

).";
