#include <numeric>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <requiredBits.h>
#include <divRoundUp.h>
#include <FunctionPrologue.h>

#include <Sintorn2/allocateHierarchy.h>
#include <Sintorn2/config.h>

using namespace ge::gl;

void sintorn2::allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2","sintorn2.method.config");

  auto cfg = *vars.get<Config>("sintorn2.method.config");

  vars.reCreate      <Buffer  >("sintorn2.method.nodePool"   ,cfg.nodesSize                   );
  vars.reCreate      <Buffer  >("sintorn2.method.aabbPool"   ,cfg.aabbsSize                   );
  vars.reCreate      <Buffer  >("sintorn2.method.levelNodeCounter",cfg.nofLevels*sizeof(uint32_t)*4);

  
  uint32_t nnodes = 0;
  for(auto const&x:cfg.nofNodesPerLevel)
    nnodes += x;
  vars.reCreate      <Buffer  >("sintorn2.method.activeNodes",nnodes*sizeof(uint32_t));

  vars.reCreate<Buffer>("sintorn2.method.debugBuffer",sizeof(uint32_t)*nnodes);


  vars.reCreate      <Buffer  >("sintorn2.method.nodeCounter",sizeof(uint32_t)*cfg.clustersX*cfg.clustersY);
}
