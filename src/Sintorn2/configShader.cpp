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



const uint warpBits        = uint(ceil(log2(float(WARP))));
const uint clustersX       = uint(WINDOW_X/TILE_X) + uint(WINDOW_X%TILE_X != 0u);
const uint clustersY       = uint(WINDOW_Y/TILE_Y) + uint(WINDOW_Y%TILE_Y != 0u);
const uint xBits           = uint(ceil(log2(float(clustersX))));
const uint yBits           = uint(ceil(log2(float(clustersY))));
const uint zBits           = MIN_Z_BITS>0?MIN_Z_BITS:max(max(xBits,yBits),MIN_Z_BITS);
const uint allBits         = xBits + yBits + zBits;
const uint nofLevels       = uint(allBits/warpBits) + uint(allBits%warpBits != 0u);
const uint uintsPerWarp    = uint(WARP/32u);

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
