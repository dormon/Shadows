#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>

#include <RSSV/propagateAABB.h>
#include <RSSV/createPropagateAABBProgram.h>
#include <RSSV/config.h>

using namespace ge::gl;

void rssv::propagateAABB(vars::Vars&vars){
  FUNCTION_CALLER();
  createPropagateAABBProgram(vars);

  auto prg = vars.get<Program>("rssv.method.propagateAABBProgram");

  auto const cfg        = *vars.get<Config>("rssv.method.config");
  auto levelNodeCounter =  vars.get<Buffer >("rssv.method.levelNodeCounter");
  auto activeNodes      =  vars.get<Buffer >("rssv.method.activeNodes");


  auto hierarchy = vars.get<Buffer>("rssv.method.hierarchy");
  hierarchy->bindBase(GL_SHADER_STORAGE_BUFFER,0);

  activeNodes     ->bindBase(GL_SHADER_STORAGE_BUFFER,4);
  levelNodeCounter->bind    (GL_DISPATCH_INDIRECT_BUFFER);
  levelNodeCounter->bindBase(GL_SHADER_STORAGE_BUFFER,3);

  prg->use();


#if 0
  auto debugBuffer =  vars.get<Buffer >("rssv.method.debugBuffer");
  debugBuffer->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  debugBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,7);
#endif


  for(int32_t level=cfg.nofLevels-2;level>=0;--level){
    prg->set1ui("destLevel",level);
    glDispatchComputeIndirect(((level)*4u)*sizeof(uint32_t));
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);
  }

#if 0
  std::vector<uint32_t>debugData;
  debugBuffer->getData(debugData);

  for(uint32_t i=0;i<100;++i)
    std::cerr << debugData[i] << std::endl;
  exit(1);
#endif

}
