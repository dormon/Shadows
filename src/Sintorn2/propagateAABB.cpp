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
  auto levelNodeCounter =  vars.get<Buffer >("sintorn2.method.levelNodeCounter");
  auto activeNodes      =  vars.get<Buffer >("sintorn2.method.activeNodes");


  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);

  prg->use();

  //prg->set1ui("destLevel",cfg.nofLevels-2);
  //glDispatchCompute(divRoundUp(cfg.nofNodesPerLevel[cfg.nofLevels-2],1024),1024,1);
  //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

#if 0
  auto debugBuffer =  vars.get<Buffer >("sintorn2.method.debugBuffer");
  debugBuffer->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  debugBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,7);
#endif

  levelNodeCounter->bind(GL_DISPATCH_INDIRECT_BUFFER);
  activeNodes->bindBase(GL_SHADER_STORAGE_BUFFER,4);
  prg->set1ui("destLevel",cfg.nofLevels-2);
  glDispatchComputeIndirect(((cfg.nofLevels-2)*4u)*sizeof(uint32_t));
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

#if 0
  std::vector<uint32_t>debugData;
  debugBuffer->getData(debugData);

  for(uint32_t i=0;i<100;++i)
    std::cerr << debugData[i] << std::endl;
  exit(1);
#endif


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
