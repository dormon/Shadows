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
      ,"rssv.param.tileX"
      ,"rssv.param.tileY"
      );

  auto const wavefrontSize =  vars.getSizeT("wavefrontSize");
  auto const windowSize    = *vars.get<glm::uvec2>("windowSize");
  auto const minZBits      =  vars.getUint32("rssv.param.minZBits");
  auto const tileX         =  vars.getUint32("rssv.param.tileX");
  auto const tileY         =  vars.getUint32("rssv.param.tileY");

  vars.reCreate<Config>("rssv.method.config",wavefrontSize,windowSize.x,windowSize.y,tileX,tileY,minZBits);
}
