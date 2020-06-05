#include <RSSV/getAABBShader.h>

std::string const rssv::getAABBShaderFWD = R".(
void loadAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node,in uint lii);
void computeAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node,in uint lii);
void getAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node,in uint lii);
vec3 getAABBCenter(in uint level,in uint node,in uint lii);
).";

std::string const rssv::getAABBShader = R".(

void loadAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node,in uint lii){
#if MEMORY_OPTIM == 1
  uint w = aabbPointer[1 + nodeLevelOffset[level] + node*WARP + lii ];
  minCorner[0] = aabbPool[w*floatsPerAABB + 0u];
  minCorner[1] = aabbPool[w*floatsPerAABB + 2u];
  minCorner[2] = aabbPool[w*floatsPerAABB + 4u];
  maxCorner[0] = aabbPool[w*floatsPerAABB + 1u];
  maxCorner[1] = aabbPool[w*floatsPerAABB + 3u];
  maxCorner[2] = aabbPool[w*floatsPerAABB + 5u];
#else
  minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + (node*WARP + lii)*floatsPerAABB + 0u];
  minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + (node*WARP + lii)*floatsPerAABB + 2u];
  minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + (node*WARP + lii)*floatsPerAABB + 4u];
  maxCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + (node*WARP + lii)*floatsPerAABB + 1u];
  maxCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + (node*WARP + lii)*floatsPerAABB + 3u];
  maxCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + (node*WARP + lii)*floatsPerAABB + 5u];
#endif
}

void computeAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node,in uint lii){
  uvec3 xyz = (demorton(((node<<warpBits)+lii)<<(warpBits*(nofLevels-1-level))));

  float startX = -1.f + xyz.x*levelTileSizeClipSpace[nofLevels-1].x;
  float startY = -1.f + xyz.y*levelTileSizeClipSpace[nofLevels-1].y;
  float endX   = min(startX + levelTileSizeClipSpace[level].x,1.f);
  float endY   = min(startY + levelTileSizeClipSpace[level].y,1.f);
  float startZ = Z_TO_DEPTH(CLUSTER_TO_Z(xyz.z                             ));
  float endZ   = Z_TO_DEPTH(CLUSTER_TO_Z(xyz.z+(1u<<levelTileBits[level].z)));

  minCorner[0] = startX;
  minCorner[1] = startY;
  minCorner[2] = startZ;

  maxCorner[0] = endX;
  maxCorner[1] = endY;
  maxCorner[2] = endZ;
}

void getAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node,in uint lii){
#if NO_AABB == 1
  computeAABB(minCorner,maxCorner,level,node,lii);
#else
  loadAABB(minCorner,maxCorner,level,node,lii);
#endif
}

vec3 getAABBCenter(in uint level,in uint node,in uint lii){
  vec3 minCorner;
  vec3 maxCorner;
  getAABB(minCorner,maxCorner,level,node,lii);
  return (minCorner+maxCorner)*.5f;
}

).";
