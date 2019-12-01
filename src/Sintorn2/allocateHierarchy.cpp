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

  auto const wavefrontSize =  vars.getSizeT("wavefrontSize");
  auto const windowSize    = *vars.get<glm::uvec2>("windowSize");
  auto const minZ          =  vars.getUint32("sintorn2.param.minZBits");
  auto const tileX         =  vars.getUint32("sintorn2.param.tileX");
  auto const tileY         =  vars.getUint32("sintorn2.param.tileY");

  uint32_t const warpBits      = requiredBits(wavefrontSize);
  uint32_t const clustersX     = divRoundUp(windowSize.x,tileX);
  uint32_t const clustersY     = divRoundUp(windowSize.y,tileY);
  uint32_t const xBits         = requiredBits(clustersX);
  uint32_t const yBits         = requiredBits(clustersY);
  uint32_t const zBits         = minZ>0?minZ:glm::max(glm::max(xBits,yBits),minZ);
  uint32_t const clustersZ     = 1 << zBits;
  uint32_t const allBits       = xBits + yBits + zBits;
  uint32_t const nofLevels     = divRoundUp(allBits,warpBits);
  uint32_t const uintsPerWarp  = wavefrontSize / (sizeof(uint32_t)*8);


#define PRINT(x) std::cerr << #x ": " << x << std::endl
  /*

  PRINT(warpBits      );
  PRINT(warpX         );
  PRINT(clustersX     );
  PRINT(clustersY     );
  PRINT(xBits         );
  PRINT(yBits         );
  PRINT(zBits         );
  PRINT(clustersZ     );
  PRINT(allBits       );
  PRINT(nofLevels     );
  PRINT(uintsPerWarp  );

  PRINT(uintsPerNode  );
  */

  //nodes for 4 levels
  //n      = 1 + 32 + 32*32 + 32*32*32
  //
  //n*32+1 = 1 + 32 + 32*32 + 32*32*32 + 32*32*32*32
  //n*32+1 - n = 32^L
  //
  //n*(32-1) + 1 = 32^L
  //
  //n = (32^L - 1) / (32-1)
  

  int32_t bits = allBits;
  uint32_t nofNodes = 0;
  while(bits>0){
    nofNodes += 1<<bits;
    bits -= warpBits;
  }
  nofNodes += 1;

  uint32_t const floatsPerAABB = 6;

  auto const nodesSize = divRoundUp(nofNodes,wavefrontSize) * sizeof(uint32_t) * uintsPerWarp;
  
  auto const aabbSize  = nofNodes * floatsPerAABB * sizeof(float);

  vars.reCreate      <uint32_t>("sintorn2.method.allBits"    ,allBits);
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
