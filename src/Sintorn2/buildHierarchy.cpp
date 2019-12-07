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
#include <Sintorn2/configShader.h>
#include <Sintorn2/config.h>

using namespace ge::gl;

#include <iomanip>

namespace sintorn2{
  void prepareFixIndirectComputeBuffer(vars::Vars&vars){
    FUNCTION_PROLOGUE("sintorn2.method"
        ,"sintorn2.method.config"
        ,"wavefrontSize"
        );

    auto const wavefrontSize  =  vars.getSizeT           ("wavefrontSize"         );
    auto const cfg            = *vars.get<Config>        ("sintorn2.method.config");
    std::string const src = R".(
    layout(local_size_x=32)in;
    layout(binding = 3)buffer LevelNodeCounter{uint levelNodeCounter[];};
    
    void main(){
      if(gl_LocalInvocationIndex >= 4*nofLevels)return;
      uint m = uint(gl_LocalInvocationIndex) % 4u;
      if(m>0)
        levelNodeCounter[gl_LocalInvocationIndex] = 1u;
    }

    ).";
    vars.reCreate<Program>("sintorn2.method.fixIndirectComputeProgram",
        std::make_shared<Shader>(GL_COMPUTE_SHADER,
          "#version 450\n",
          Shader::define("WARP"      ,(uint32_t)wavefrontSize),
          Shader::define("WINDOW_X"  ,(uint32_t)cfg.windowX  ),
          Shader::define("WINDOW_Y"  ,(uint32_t)cfg.windowY  ),
          Shader::define("MIN_Z_BITS",(uint32_t)cfg.minZBits ),
          Shader::define("TILE_X"    ,cfg.tileX              ),
          Shader::define("TILE_Y"    ,cfg.tileY              ),
          sintorn2::configShader,
          src));
  }
  void fixIndirectComputeBuffer(vars::Vars&vars){
    prepareFixIndirectComputeBuffer(vars);
    auto levelNodeCounter =  vars.get<Buffer >("sintorn2.method.levelNodeCounter");
    auto prg              =  vars.get<Program>("sintorn2.method.fixIndirectComputeProgram");
    prg->use();
    levelNodeCounter->bindBase(GL_SHADER_STORAGE_BUFFER,3);
    glDispatchCompute(1,1,1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  }
}

void sintorn2::buildHierarchy(vars::Vars&vars){
  FUNCTION_CALLER();

  sintorn2::computeConfig(vars);
  sintorn2::allocateHierarchy(vars);
  sintorn2::createBuildHierarchyProgram(vars);
  //exit(0);
  auto depth            =  vars.get<GBuffer>("gBuffer")->depth;
  auto prg              =  vars.get<Program>("sintorn2.method.buildHierarchyProgram");
  auto nodePool         =  vars.get<Buffer >("sintorn2.method.nodePool");
  auto aabbPool         =  vars.get<Buffer >("sintorn2.method.aabbPool");
  //auto nodeCounter      =  vars.get<Buffer >("sintorn2.method.nodeCounter");
  auto levelNodeCounter =  vars.get<Buffer >("sintorn2.method.levelNodeCounter");
  auto activeNodes      =  vars.get<Buffer >("sintorn2.method.activeNodes");

#if 0
  auto debugBuffer      =  vars.get<Buffer >("sintorn2.method.debugBuffer");
  debugBuffer        ->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  debugBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,7);
#endif

  auto cfg              = *vars.get<Config >("sintorn2.method.config");

  nodePool        ->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  levelNodeCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  //nodeCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  //aabbPool   ->clear(GL_R32F ,GL_RED        ,GL_FLOAT       );

  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  //nodeCounter->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  levelNodeCounter->bindBase(GL_SHADER_STORAGE_BUFFER,3);
  activeNodes     ->bindBase(GL_SHADER_STORAGE_BUFFER,4);
  
  depth->bind(1);
  
  prg->use();
  glDispatchCompute(cfg.clustersX,cfg.clustersY,1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  fixIndirectComputeBuffer(vars);

  

#if 0
  cfg.print();
  std::vector<uint32_t>nodes;
  nodePool->getData(nodes);

  std::vector<uint32_t>lc;
  levelNodeCounter->getData(lc);
  for(uint32_t l=0;l<cfg.nofLevels;++l){
    std::cerr << "L" << l << ": ";
    for(uint32_t i=0;i<4;++i)
      std::cerr << lc[l*4+i] << " ";
    std::cerr << std::endl;
  }

  std::vector<uint32_t>an;
  activeNodes->getData(an);
  for(uint32_t l=0;l<cfg.nofLevels;++l){
    std::vector<uint32_t>ll;
    for(uint32_t i=0;i<lc[l*4];++i)
      ll.push_back(an[cfg.nodeLevelOffset[l]+i]);
    //std::sort(ll.begin(),ll.end());

    std::cerr << "L" << l << ": " << std::endl;;
    for(uint32_t i=0;i<ll.size();++i)
      std::cerr << " " << ll[i] << "-" << nodes[cfg.nodeLevelOffsetInUints[l]+ll[i]] << std::endl;
      //std::cerr << " " << ll[i] << "-" << nodes[ll[i]] << std::endl;
    std::cerr << std::endl;
  }

  exit(0);
#endif

  

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
