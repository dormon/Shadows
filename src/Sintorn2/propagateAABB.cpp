#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Sintorn2/propagateAABB.h>
#include <Sintorn2/createPropagateAABBProgram.h>

using namespace ge::gl;

void sintorn2::propagateAABB(vars::Vars&vars){
  createPropagateAABBProgram(vars);

  auto prg = vars.get<Program>("sintorn2.method.propagateAABBProgram");

  auto const warpBits  = vars.getUint32("sintorn2.method.warpBits" );
  auto const allBits   = vars.getUint32("sintorn2.method.allBits"  );
  auto const nofLevels = vars.getUint32("sintorn2.method.nofLevels");

  auto nodePool    = vars.get<Buffer >("sintorn2.method.nodePool");
  auto aabbPool    = vars.get<Buffer >("sintorn2.method.aabbPool");


  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);

  auto level = nofLevels - 2;
  auto nodesPerLevel = 1u << (uint32_t)glm::max(((int)allBits) - ((int)(warpBits * (nofLevels-level+1))),0);

  prg->use();
  prg->set1ui("destLevel",nofLevels-2);
  //glDispatchCompute(nodesPerLevel,1,1);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

}
