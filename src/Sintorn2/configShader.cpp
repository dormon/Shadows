#include <Sintorn2/configShader.h>

std::string const sintorn2::configShader = R".(

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

const uint longestAxis  = clamp(uint(uint(bitLength[2] == yBits) + uint(bitLength[2] == zBits)*2u),0u,2u);
const uint middleAxis   = clamp(uint(uint(bitLength[1] == yBits) + uint(bitLength[1] == zBits)*2u),0u,2u);
const uint shortestAxis = clamp(uint(uint(bitLength[0] == yBits) + uint(bitLength[0] == zBits)*2u),0u,2u);

#define QUANTIZE_Z(z) clamp(uint(log(-z/NEAR) / log(1.f+2.f*tan(FOVY/2.f)/clustersY)),0u,clustersZ-1u)
#define CLUSTER_TO_Z(i) (-NEAR * exp((i)*log(1.f + 2.f*tan(FOVY/2.f)/clustersY)))


#ifdef FAR_IS_INFINITE
  #define DEPTH_TO_Z(d) (2.f*NEAR    /(d - 1.f))
  #define Z_TO_DEPTH(z) ((2.f*NEAR)/z+1.f)
#else
  #define DEPTH_TO_Z(d) (2.f*NEAR*FAR/(d*(FAR-NEAR)-FAR-NEAR))
  #define Z_TO_DEPTH(z) (((2.f*NEAR*FAR/z)+FAR+NEAR)/(FAR-NEAR))
#endif

const uint twoLongest[] = {
  middleAxis ,
  longestAxis,
};

#define WHICH_BIT(b) uint(b<bitTogether[0]*3u?b%3u:b<bitTogether[0]*3u+bitTogether[1]*2u?twoLongest[b%2u]:b<allBits?longestAxis:noAxis)
#define IS_BIT(b,a)  uint(WHICH_BIT(b) == a)
#define SET_BIT(b,a) uint(IS_BIT(b,a) << b)

#line 730
const uvec3 bitPosition = uvec3(
  SET_BIT( 0u,0u)| 
  SET_BIT( 1u,0u)| 
  SET_BIT( 2u,0u)| 
  SET_BIT( 3u,0u)| 
  SET_BIT( 4u,0u)| 
  SET_BIT( 5u,0u)| 
  SET_BIT( 6u,0u)| 
  SET_BIT( 7u,0u)| 
  SET_BIT( 8u,0u)| 
  SET_BIT( 9u,0u)| 
  SET_BIT(10u,0u)| 
  SET_BIT(11u,0u)| 
  SET_BIT(12u,0u)| 
  SET_BIT(13u,0u)| 
  SET_BIT(14u,0u)| 
  SET_BIT(15u,0u)| 
  SET_BIT(16u,0u)| 
  SET_BIT(17u,0u)| 
  SET_BIT(18u,0u)| 
  SET_BIT(19u,0u)| 
  SET_BIT(20u,0u)| 
  SET_BIT(21u,0u)| 
  SET_BIT(22u,0u)| 
  SET_BIT(23u,0u)| 
  SET_BIT(24u,0u)| 
  SET_BIT(25u,0u)| 
  SET_BIT(26u,0u)| 
  SET_BIT(27u,0u)| 
  SET_BIT(28u,0u)| 
  SET_BIT(29u,0u)| 
  SET_BIT(30u,0u)| 
  SET_BIT(31u,0u),
#line 107
  SET_BIT( 0u,1u)| 
  SET_BIT( 1u,1u)| 
  SET_BIT( 2u,1u)| 
  SET_BIT( 3u,1u)| 
  SET_BIT( 4u,1u)| 
  SET_BIT( 5u,1u)| 
  SET_BIT( 6u,1u)| 
  SET_BIT( 7u,1u)| 
  SET_BIT( 8u,1u)| 
  SET_BIT( 9u,1u)| 
  SET_BIT(10u,1u)| 
  SET_BIT(11u,1u)| 
  SET_BIT(12u,1u)| 
  SET_BIT(13u,1u)| 
  SET_BIT(14u,1u)| 
  SET_BIT(15u,1u)| 
  SET_BIT(16u,1u)| 
  SET_BIT(17u,1u)| 
  SET_BIT(18u,1u)| 
  SET_BIT(19u,1u)| 
  SET_BIT(20u,1u)| 
  SET_BIT(21u,1u)| 
  SET_BIT(22u,1u)| 
  SET_BIT(23u,1u)| 
  SET_BIT(24u,1u)| 
  SET_BIT(25u,1u)| 
  SET_BIT(26u,1u)| 
  SET_BIT(27u,1u)| 
  SET_BIT(28u,1u)| 
  SET_BIT(29u,1u)| 
  SET_BIT(30u,1u)| 
  SET_BIT(31u,1u),
  SET_BIT( 0u,2u)| 
  SET_BIT( 1u,2u)| 
  SET_BIT( 2u,2u)| 
  SET_BIT( 3u,2u)| 
  SET_BIT( 4u,2u)| 
  SET_BIT( 5u,2u)| 
  SET_BIT( 6u,2u)| 
  SET_BIT( 7u,2u)| 
  SET_BIT( 8u,2u)| 
  SET_BIT( 9u,2u)| 
  SET_BIT(10u,2u)| 
  SET_BIT(11u,2u)| 
  SET_BIT(12u,2u)| 
  SET_BIT(13u,2u)| 
  SET_BIT(14u,2u)| 
  SET_BIT(15u,2u)| 
  SET_BIT(16u,2u)| 
  SET_BIT(17u,2u)| 
  SET_BIT(18u,2u)| 
  SET_BIT(19u,2u)| 
  SET_BIT(20u,2u)| 
  SET_BIT(21u,2u)| 
  SET_BIT(22u,2u)| 
  SET_BIT(23u,2u)| 
  SET_BIT(24u,2u)| 
  SET_BIT(25u,2u)| 
  SET_BIT(26u,2u)| 
  SET_BIT(27u,2u)| 
  SET_BIT(28u,2u)| 
  SET_BIT(29u,2u)| 
  SET_BIT(30u,2u)| 
  SET_BIT(31u,2u)
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
  1u << uint(max(int(allBits) - int((nofLevels-1u)*warpBits),0)),
  1u << uint(max(int(allBits) - int((nofLevels-2u)*warpBits),0)),
  1u << uint(max(int(allBits) - int((nofLevels-3u)*warpBits),0)),
  1u << uint(max(int(allBits) - int((nofLevels-4u)*warpBits),0)),
  1u << uint(max(int(allBits) - int((nofLevels-5u)*warpBits),0)),
  1u << uint(max(int(allBits) - int((nofLevels-6u)*warpBits),0)),
};
#line 70

const uint nodeLevelOffset[6] = {
  0,
  0 + nodesPerLevel[0],
  0 + nodesPerLevel[0] + nodesPerLevel[1],
  0 + nodesPerLevel[0] + nodesPerLevel[1] + nodesPerLevel[2],
  0 + nodesPerLevel[0] + nodesPerLevel[1] + nodesPerLevel[2] + nodesPerLevel[3],
  0 + nodesPerLevel[0] + nodesPerLevel[1] + nodesPerLevel[2] + nodesPerLevel[3] + nodesPerLevel[4],
};

const uint nodeLevelSizeInUints[6] = {
  max(nodesPerLevel[0] >> warpBits,1u) * uintsPerWarp,
  max(nodesPerLevel[1] >> warpBits,1u) * uintsPerWarp,
  max(nodesPerLevel[2] >> warpBits,1u) * uintsPerWarp,
  max(nodesPerLevel[3] >> warpBits,1u) * uintsPerWarp,
  max(nodesPerLevel[4] >> warpBits,1u) * uintsPerWarp,
  max(nodesPerLevel[5] >> warpBits,1u) * uintsPerWarp,
};

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

const uint aabbLevelOffsetInFloats[6] = {
  0,
  0 + aabbLevelSizeInFloats[0],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2] + aabbLevelSizeInFloats[3],
  0 + aabbLevelSizeInFloats[0] + aabbLevelSizeInFloats[1] + aabbLevelSizeInFloats[2] + aabbLevelSizeInFloats[3] + aabbLevelSizeInFloats[4],
};


).";
