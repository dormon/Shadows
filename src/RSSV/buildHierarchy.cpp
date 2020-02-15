#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <perfCounters.h>

#include <RSSV/buildHierarchy.h>
#include <RSSV/allocateHierarchy.h>
#include <RSSV/createBuildHierarchyProgram.h>
#include <RSSV/propagateAABB.h>
#include <RSSV/computeConfig.h>
#include <RSSV/configShader.h>
#include <RSSV/config.h>

using namespace ge::gl;

#include <iomanip>

void rssv::buildHierarchy(vars::Vars&vars){
  FUNCTION_CALLER();
  rssv::computeConfig(vars);
  rssv::allocateHierarchy(vars);
  rssv::createBuildHierarchyProgram(vars);
  //exit(0);
  auto gBuffer           =  vars.get<GBuffer>("gBuffer"                          );
  auto prg               =  vars.get<Program>("rssv.method.buildHierarchyProgram");
  auto levelNodeCounter  =  vars.get<Buffer >("rssv.method.levelNodeCounter"     );
  auto activeNodes       =  vars.get<Buffer >("rssv.method.activeNodes"          );
  auto discardBackfacing =  vars.getUint32   ("rssv.param.discardBackfacing"     );
  auto mergedBuffers     =  vars.getInt32    ("rssv.param.mergedBuffers"         );

  auto cfg               = *vars.get<Config >("rssv.method.config"               );

  auto depth             =  gBuffer->depth;

  if(mergedBuffers){
    auto hierarchy = vars.get<Buffer>("rssv.method.hierarchy");
    //clear node pool
    ge::gl::glClearNamedBufferSubData(hierarchy->getId(),GL_R32UI,cfg.nodeBufferOffsetInHierarchy,cfg.nodeBufferSize,GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
    //clear aabbPointer
    if(cfg.memoryOptim){
      ge::gl::glClearNamedBufferSubData(hierarchy->getId(),GL_R32UI,cfg.aabbPointerBufferOffsetInHierarchy,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
    }
    hierarchy->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  }else{
    auto nodePool = vars.get<Buffer>("rssv.method.nodePool");
    auto aabbPool = vars.get<Buffer>("rssv.method.aabbPool");
    nodePool->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
    nodePool->bindBase(GL_SHADER_STORAGE_BUFFER,0);
    aabbPool->bindBase(GL_SHADER_STORAGE_BUFFER,1);

    if(cfg.memoryOptim){
      auto aabbPointer = vars.get<Buffer>("rssv.method.aabbPointer");
      aabbPointer->bindBase(GL_SHADER_STORAGE_BUFFER,5);
      ge::gl::glClearNamedBufferSubData(aabbPointer->getId(),GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
    }
  }

  uint32_t dci[4] = {0,1,1,0};
  levelNodeCounter->clear(GL_RGBA32UI,GL_RGBA_INTEGER,GL_UNSIGNED_INT,dci);

  levelNodeCounter->bindBase(GL_SHADER_STORAGE_BUFFER,3);
  activeNodes     ->bindBase(GL_SHADER_STORAGE_BUFFER,4);
  
  depth->bind(1);

  
  prg->use();

  if(discardBackfacing){
    auto normal              =  gBuffer->normal;
    auto const lightPosition = *vars.get<glm::vec4>("rssv.method.lightPosition"   );
    normal->bind(2);
    prg->set4fv("lightPosition",glm::value_ptr(lightPosition));
  }

  if(vars.addOrGetBool("rssv.method.perfCounters.buildHierarchy")){
    perf::printComputeShaderProf([&](){
    glDispatchCompute(cfg.clustersX,cfg.clustersY,1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);
    });
  }else{
    glDispatchCompute(cfg.clustersX,cfg.clustersY,1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);
  }

  propagateAABB(vars);

}
