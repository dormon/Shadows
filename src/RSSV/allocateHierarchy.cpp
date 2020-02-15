#include <numeric>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <requiredBits.h>
#include <divRoundUp.h>
#include <FunctionPrologue.h>

#include <RSSV/allocateHierarchy.h>
#include <RSSV/config.h>

using namespace ge::gl;

void rssv::allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"rssv.method.config"
      ,"rssv.param.mergedBuffers"
      );

  auto cfg           = *vars.get<Config>("rssv.method.config"      );
  auto mergedBuffers =  vars.getInt32   ("rssv.param.mergedBuffers");

  vars.reCreate<Buffer >("rssv.method.levelNodeCounter",cfg.nofLevels*sizeof(uint32_t)*4               );
  vars.reCreate<Buffer >("rssv.method.activeNodes"     ,cfg.nofNodes *sizeof(uint32_t)                 );
  vars.reCreate<Buffer >("rssv.method.debugBuffer"     ,cfg.nofNodes *sizeof(uint32_t)                 );
  vars.reCreate<Buffer >("rssv.method.bridges"         ,cfg.nofNodes *sizeof(int32_t)                  );
  vars.reCreate<Texture>("rssv.method.stencil"         ,GL_TEXTURE_2D,GL_R32I,1,cfg.windowX,cfg.windowY);

  if(cfg.memoryOptim){
    vars.reCreate<Buffer>("rssv.method.aabbPointer",sizeof(uint32_t)*(1+cfg.nofNodes));
  }else{
    vars.erase           ("rssv.method.aabbPointer"              );
  }

  if(mergedBuffers){
    vars.reCreate<Buffer>("rssv.method.hierarchy"  ,0
        +cfg.nodeBufferSize
        +cfg.aabbBufferSize
        );
  }else{
    vars.reCreate<Buffer>("rssv.method.aabbPool"   ,cfg.aabbBufferSize);
    vars.reCreate<Buffer>("rssv.method.nodePool"   ,cfg.nodeBufferSize);
  }

  size_t sss;

  if(mergedBuffers){
    sss = 
    vars.get<Buffer>("rssv.method.hierarchy")->getSize() + 
    vars.get<Buffer>("rssv.method.activeNodes")->getSize();
  }else{
    sss =
    vars.get<Buffer>("rssv.method.nodePool")->getSize() + 
    vars.get<Buffer>("rssv.method.aabbPool")->getSize() + 
    vars.get<Buffer>("rssv.method.activeNodes")->getSize();
  }


  if(cfg.memoryOptim)
    sss += vars.get<Buffer>("rssv.method.aabbPointer")->getSize();

  GLint b;
  ge::gl::glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&b);
  std::cerr << "GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS: " << b << std::endl;

  ge::gl::glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS ,&b);
  std::cerr << "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS: " << b << std::endl;

  std::cerr << "HIERARCHY SIZE: " << (float)sss / (float)(1024*1024) << " MB" << std::endl;
}
