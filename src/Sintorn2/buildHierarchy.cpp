#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>

#include <Sintorn2/buildHierarchy.h>
#include <Sintorn2/allocateHierarchy.h>
#include <Sintorn2/createBuildHierarchyProgram.h>
#include <Sintorn2/propagateAABB.h>
#include <Sintorn2/computeConfig.h>
#include <Sintorn2/config.h>

using namespace ge::gl;

#include <iomanip>

void sintorn2::buildHierarchy(vars::Vars&vars){
  FUNCTION_CALLER();

  sintorn2::computeConfig(vars);
  sintorn2::allocateHierarchy(vars);
  sintorn2::createBuildHierarchyProgram(vars);
  //exit(0);
  auto depth       =  vars.get<GBuffer>("gBuffer")->depth;
  auto prg         =  vars.get<Program>("sintorn2.method.buildHierarchyProgram");
  auto nodePool    =  vars.get<Buffer >("sintorn2.method.nodePool");
  auto aabbPool    =  vars.get<Buffer >("sintorn2.method.aabbPool");
  auto cfg         = *vars.get<Config>("sintorn2.method.config");

  nodePool   ->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  aabbPool   ->clear(GL_R32F ,GL_RED        ,GL_FLOAT       );

  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  
  depth->bind(1);
  
  prg->use();
  glDispatchCompute(cfg.clustersX,cfg.clustersY,1);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  //propagateAABB(vars);

  //std::vector<float>d;
  //aabbPool->getData(d);
  //cfg.print();
  //for(size_t i=cfg.aabbLevelOffsetInFloats[cfg.nofLevels-2];i<cfg.aabbLevelOffsetInFloats[cfg.nofLevels-2]+10000*6;++i)
  //  std::cerr << d[i] << std::endl;

  //exit(0);


}
