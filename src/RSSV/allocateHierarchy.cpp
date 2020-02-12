#include <numeric>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <requiredBits.h>
#include <divRoundUp.h>
#include <FunctionPrologue.h>

#include <RSSV/allocateHierarchy.h>
#include <RSSV/config.h>

using namespace ge::gl;

void rssv::allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.method.config"
      ,"rssv.param.memoryOptim"
      ,"rssv.param.memoryFactor"
      );

  auto cfg          = *vars.get<Config>("rssv.method.config"     );
  auto memoryOptim  =  vars.getInt32   ("rssv.param.memoryOptim" );
  auto memoryFactor =  vars.getUint32  ("rssv.param.memoryFactor"); 

  vars.reCreate<Buffer >("rssv.method.nodePool"        ,cfg.nodesSize                                  );
  vars.reCreate<Buffer >("rssv.method.levelNodeCounter",cfg.nofLevels*sizeof(uint32_t)*4               );
  vars.reCreate<Buffer >("rssv.method.activeNodes"     ,cfg.nofNodes *sizeof(uint32_t)                 );
  vars.reCreate<Buffer >("rssv.method.debugBuffer"     ,cfg.nofNodes *sizeof(uint32_t)                 );
  vars.reCreate<Buffer >("rssv.method.bridges"         ,cfg.nofNodes *sizeof(int32_t)                  );
  vars.reCreate<Texture>("rssv.method.stencil"         ,GL_TEXTURE_2D,GL_R32I,1,cfg.windowX,cfg.windowY);


  if(memoryOptim){
    vars.reCreate<Buffer>("rssv.method.aabbPointer",sizeof(uint32_t)*(1+cfg.nofNodes)       );
    vars.reCreate<Buffer>("rssv.method.aabbPool"   ,sizeof(float)*cfg.floatsPerAABB*cfg.clustersX*cfg.clustersY*memoryFactor);
  }else{
    vars.erase           ("rssv.method.aabbPointer"              );
    vars.reCreate<Buffer>("rssv.method.aabbPool"   ,cfg.aabbsSize);
  }

  size_t sss =
  vars.get<Buffer>("rssv.method.nodePool")->getSize() + 
  vars.get<Buffer>("rssv.method.aabbPool")->getSize() + 
  vars.get<Buffer>("rssv.method.activeNodes")->getSize();

  if(memoryOptim)
    sss += vars.get<Buffer>("rssv.method.aabbPointer")->getSize();

  std::cerr << "HIERARCHY SIZE: " << (float)sss / (float)(1024*1024) << " MB" << std::endl;
}
