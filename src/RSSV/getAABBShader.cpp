#include <RSSV/getAABBShader.h>

std::string const rssv::getAABBShaderFWD = R".(
void loadAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node);
void computeAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node);
void getAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node);
vec3 getAABBCenter(in uint level,in uint node);
).";

std::string const rssv::getAABBShader = R".(

void loadAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node){
#if MEMORY_OPTIM == 1
  uint w = aabbPointer[1 + nodeLevelOffset[level] + node];
  minCorner[0] = aabbPool[w*floatsPerAABB + 0u];
  minCorner[1] = aabbPool[w*floatsPerAABB + 2u];
  minCorner[2] = aabbPool[w*floatsPerAABB + 4u];
  maxCorner[0] = aabbPool[w*floatsPerAABB + 1u];
  maxCorner[1] = aabbPool[w*floatsPerAABB + 3u];
  maxCorner[2] = aabbPool[w*floatsPerAABB + 5u];
#else
  minCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*floatsPerAABB + 0u];
  minCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*floatsPerAABB + 2u];
  minCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*floatsPerAABB + 4u];
  maxCorner[0] = aabbPool[aabbLevelOffsetInFloats[level] + node*floatsPerAABB + 1u];
  maxCorner[1] = aabbPool[aabbLevelOffsetInFloats[level] + node*floatsPerAABB + 3u];
  maxCorner[2] = aabbPool[aabbLevelOffsetInFloats[level] + node*floatsPerAABB + 5u];
#endif
}

void computeAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node){
  uvec3 xyz = (demorton(node<<(warpBits*(nofLevels-1-level))));

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

void getAABB(out vec3 minCorner,out vec3 maxCorner,in uint level,in uint node){
#if NO_AABB == 1
  computeAABB(minCorner,maxCorner,level,node);
#else
  loadAABB(minCorner,maxCorner,level,node);
#endif
}

vec3 getAABBCenter(in uint level,in uint node){
#if USE_BRIDGE_POOL == 1
  vec3 res;
#if MEMORY_OPTIM == 1
  uint w = aabbPointer[1 + nodeLevelOffset[level] + node];
  res[0] = bridgePool[w*floatsPerBridge + 0u];
  res[1] = bridgePool[w*floatsPerBridge + 1u];
  res[2] = bridgePool[w*floatsPerBridge + 2u];
#else
  res[0] = bridgePool[bridgeLevelOffsetInFloats[level] + node*floatsPerBridge + 0u];
  res[1] = bridgePool[bridgeLevelOffsetInFloats[level] + node*floatsPerBridge + 1u];
  res[2] = bridgePool[bridgeLevelOffsetInFloats[level] + node*floatsPerBridge + 2u];
#endif
  return res;


#else
  vec3 minCorner;
  vec3 maxCorner;
  getAABB(minCorner,maxCorner,level,node);
  return (minCorner+maxCorner)*.5f;
#endif
}

).";
