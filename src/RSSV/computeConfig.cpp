#include <glm/glm.hpp>

#include <Vars/Vars.h>

#include <FunctionPrologue.h>
#include <requiredBits.h>
#include <divRoundUp.h>

#include <RSSV/computeConfig.h>
#include <RSSV/config.h>

void rssv::computeConfig(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"wavefrontSize"
      ,"windowSize"
      ,"rssv.param.minZBits"
      ,"rssv.param.memoryOptim"
      ,"rssv.param.memoryFactor"
      );

  auto const wavefrontSize =  vars.getSizeT       ("wavefrontSize"          );
  auto const windowSize    = *vars.get<glm::uvec2>("windowSize"             );
  auto const minZBits      =  vars.getUint32      ("rssv.param.minZBits"    );
  auto const memoryOptim   =  vars.getInt32       ("rssv.param.memoryOptim" );
  auto const memoryFactor  =  vars.getInt32       ("rssv.param.memoryFactor");

  vars.reCreate<Config>("rssv.method.config"
      ,wavefrontSize
      ,windowSize.x
      ,windowSize.y
      ,minZBits
      ,memoryOptim
      ,memoryFactor
      );
}
