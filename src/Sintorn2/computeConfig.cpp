#include <glm/glm.hpp>

#include <Vars/Vars.h>

#include <FunctionPrologue.h>
#include <requiredBits.h>
#include <divRoundUp.h>

#include <Sintorn2/computeConfig.h>
#include <Sintorn2/config.h>

void sintorn2::computeConfig(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method"
      ,"wavefrontSize"
      ,"windowSize"
      ,"sintorn2.param.minZBits"
      ,"sintorn2.param.tileX"
      ,"sintorn2.param.tileY"
      );

  auto const wavefrontSize =  vars.getSizeT("wavefrontSize");
  auto const windowSize    = *vars.get<glm::uvec2>("windowSize");
  auto const minZBits      =  vars.getUint32("sintorn2.param.minZBits");
  auto const tileX         =  vars.getUint32("sintorn2.param.tileX");
  auto const tileY         =  vars.getUint32("sintorn2.param.tileY");

  vars.reCreate<Config>("sintorn2.method.config", uint32_t(wavefrontSize),windowSize.x,windowSize.y,tileX,tileY,minZBits);
}
