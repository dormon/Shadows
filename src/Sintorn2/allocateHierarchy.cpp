#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <requiredBits.h>
#include <divRoundUp.h>
#include <FunctionPrologue.h>

#include <Sintorn2/allocateHierarchy.h>

using namespace ge::gl;

// hierarchy - nodePool and AABBs
// node{
//   uint mask     [uintsPerWarp]  ;
//   uint ptrToAABB                ;
//   uint padding  [uintsPerWarp-1];        
// };
//
// WARP = 32
// |Level0|Level1                         |Level2
// | root | |child0| |child1| ... |child31| |child0ofChild0| |child1ofChild0| ...
// 
// AABB{
//   float x[2];
//   float y[2];
//   float z[2];
// };
//
void sintorn2::allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2","windowSize","wavefrontSize","sintorn2.param.minZBits");

  auto wavefrontSize =  vars.getSizeT("wavefrontSize");
  auto windowSize    = *vars.get<glm::uvec2>("windowSize");
  auto minZ          =  vars.getUint32("sintorn2.param.minZBits");

  uint32_t const warpBits      = requiredBits(wavefrontSize);
  uint32_t const warpBitsX     = divRoundUp(warpBits,2u);
  uint32_t const warpBitsY     = warpBits-warpBitsX;
  uint32_t const warpX         = 1u<<warpBitsX;
  uint32_t const warpY         = 1u<<warpBitsY;
  uint32_t const clustersX     = divRoundUp(windowSize.x,warpX);
  uint32_t const clustersY     = divRoundUp(windowSize.y,warpY);
  uint32_t const xBits         = requiredBits(clustersX);
  uint32_t const yBits         = requiredBits(clustersY);
  uint32_t const zBits         = minZ>0?minZ:glm::max(glm::max(xBits,yBits),minZ);
  uint32_t const clustersZ     = 1 << zBits;
  uint32_t const allBits       = xBits + yBits + zBits;
  uint32_t const nofLevels     = divRoundUp(allBits,warpBits);
  uint32_t const uintsPerWarp  = wavefrontSize / (sizeof(uint32_t)*8);

  uint32_t const uintsPerNode  = uintsPerWarp + 1 + (uintsPerNode-1);

  //nodes for 4 levels
  //n      = 1 + 32 + 32*32 + 32*32*32
  //
  //n*32+1 = 1 + 32 + 32*32 + 32*32*32 + 32*32*32*32
  //n*32+1 - n = 32^L
  //
  //n*(32-1) + 1 = 32^L
  //
  //n = (32^L - 1) / (32-1)
  
  uint32_t const nofNodes = (glm::pow(wavefrontSize,nofLevels) - 1) / (wavefrontSize - 1);


  uint32_t const floatsPerAABB = 6;

  auto const nodesSize = nofNodes * uintsPerNode * sizeof(uint32_t);
  auto const aabbSize  = nofNodes * floatsPerAABB * sizeof(float);

  vars.reCreate      <uint32_t>("sintorn2.method.allBits"    ,allBits);
  vars.reCreate      <uint32_t>("sintorn2.method.warpBitsX"  ,warpBitsX);
  vars.reCreate      <uint32_t>("sintorn2.method.warpBitsY"  ,warpBitsY);
  vars.reCreate      <uint32_t>("sintorn2.method.warpBits"   ,warpBits );
  vars.reCreate      <uint32_t>("sintorn2.method.xBits"      ,xBits    );
  vars.reCreate      <uint32_t>("sintorn2.method.yBits"      ,yBits    );
  vars.reCreate      <uint32_t>("sintorn2.method.zBits"      ,zBits    );
  vars.reCreate      <uint32_t>("sintorn2.method.clustersX"  ,clustersX);
  vars.reCreate      <uint32_t>("sintorn2.method.clustersY"  ,clustersY);
  vars.reCreate      <Buffer  >("sintorn2.method.nodePool"   ,nodesSize );
  vars.reCreate      <Buffer  >("sintorn2.method.aabbPool"   ,aabbSize  );
  vars.reCreate      <Buffer  >("sintorn2.method.aabbCounter",sizeof(uint32_t));
}
