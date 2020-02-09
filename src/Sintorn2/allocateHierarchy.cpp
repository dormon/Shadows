#include <numeric>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <requiredBits.h>
#include <divRoundUp.h>
#include <FunctionPrologue.h>

#include <Sintorn2/allocateHierarchy.h>
#include <Sintorn2/config.h>

using namespace ge::gl;

void sintorn2::allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method"
      ,"sintorn2.method.config"
      ,"sintorn2.param.memoryOptim"
      );

  auto cfg         = *vars.get<Config>("sintorn2.method.config"     );
  auto memoryOptim =  vars.getInt32   ("sintorn2.param.memoryOptim" );

  vars.reCreate<Buffer>("sintorn2.method.nodePool"        ,cfg.nodesSize                   );
  vars.reCreate<Buffer>("sintorn2.method.levelNodeCounter",cfg.nofLevels*sizeof(uint32_t)*4);
  vars.reCreate<Buffer>("sintorn2.method.activeNodes"     ,cfg.nofNodes *sizeof(uint32_t)  );
  vars.reCreate<Buffer>("sintorn2.method.debugBuffer"     ,cfg.nofNodes *sizeof(uint32_t)  );

  if(memoryOptim){
    vars.reCreate<Buffer>("sintorn2.method.aabbPointer",sizeof(uint32_t)*(1+cfg.nofNodes)  );
    vars.reCreate<Buffer>("sintorn2.method.aabbPool"   ,cfg.aabbsSize                      );
  }else{
    vars.erase           ("sintorn2.method.aabbPointer"              );
    vars.reCreate<Buffer>("sintorn2.method.aabbPool"   ,cfg.aabbsSize);
  }


  std::cerr <<  "nofNodes: " << cfg.nofNodes << std::endl;
  std::cerr <<  "aabbSize: " << cfg.aabbsSize << std::endl;
  cfg.print();
  size_t sss =
  vars.get<Buffer>("sintorn2.method.nodePool")->getSize() + 
  vars.get<Buffer>("sintorn2.method.aabbPool")->getSize() + 
  vars.get<Buffer>("sintorn2.method.activeNodes")->getSize();
  std::cerr << "HIERARCHY SIZE: " << (float)sss / (float)(1024*1024) << " MB" << std::endl;
}
