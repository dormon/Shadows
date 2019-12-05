#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>

#include <Sintorn2/propagateAABB.h>
#include <Sintorn2/createPropagateAABBProgram.h>
#include <Sintorn2/config.h>

using namespace ge::gl;

void sintorn2::propagateAABB(vars::Vars&vars){
  FUNCTION_CALLER();
  createPropagateAABBProgram(vars);

  auto prg = vars.get<Program>("sintorn2.method.propagateAABBProgram");

  auto const cfg       =*vars.get<Config>("sintorn2.method.config");

  auto nodePool    = vars.get<Buffer >("sintorn2.method.nodePool");
  auto aabbPool    = vars.get<Buffer >("sintorn2.method.aabbPool");


  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);

  auto level = cfg.nofLevels - 2;
  //auto level = 0;

  prg->use();

  //prg->set1ui("destLevel",cfg.nofLevels-2);
  //glDispatchCompute(divRoundUp(cfg.nofNodesPerLevel[cfg.nofLevels-2],1024),1024,1);
  //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

/*
  for(int32_t level=cfg.nofLevels-2;level>=0;--level){
    prg->set1ui("destLevel",level);
    if(cfg.nofNodesPerLevel[level]>1024)
      glDispatchCompute(divRoundUp(cfg.nofNodesPerLevel[level],1024),1024,1);
    else
      glDispatchCompute(cfg.nofNodesPerLevel[level],1,1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  }
// */

}
