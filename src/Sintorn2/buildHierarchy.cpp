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
  auto nodeCounter =  vars.get<Buffer >("sintorn2.method.nodeCounter");
  auto cfg         = *vars.get<Config>("sintorn2.method.config");

  nodePool   ->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  nodeCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  //aabbPool   ->clear(GL_R32F ,GL_RED        ,GL_FLOAT       );

  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  nodeCounter->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  
  depth->bind(1);
  
  prg->use();
  glDispatchCompute(cfg.clustersX,cfg.clustersY,1);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  propagateAABB(vars);

  /*
  std::vector<uint32_t>d;
  nodePool->getData(d);
  cfg.print();
  auto level = cfg.nofLevels-1;
  //level = 0;
  for(size_t i=cfg.nodeLevelOffsetInUints[level];i<cfg.nodeLevelOffsetInUints[level]+cfg.nodeLevelSizeInUints[level];++i)
    std::cerr << d[i] << std::endl;

  // */

  /*
  std::vector<float>d;
  aabbPool->getData(d);
  cfg.print();
  auto level = cfg.nofLevels-2;
  //level = 0;
  for(size_t i=cfg.aabbLevelOffsetInFloats[level];i<cfg.aabbLevelOffsetInFloats[level]+cfg.aabbLevelSizeInFloats[level];i+=6){
    if(
        d[i+0] == 0 && 
        d[i+1] == 0 &&
        d[i+2] == 0 &&
        d[i+3] == 0 &&
        d[i+4] == 0 &&
        d[i+5] == 0 )continue;
    std::cerr << "node " << i/6 << " : ";
    for(int j=0;j<6;++j)
      std::cerr << d[i+j] << " ";
    std::cerr << std::endl;
  }

  // */

  //exit(0);


}
