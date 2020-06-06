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
      ,"rssv.param.useBridgePool"
      ,"rssv.param.scaledQuantization"
      ,"args.camera.near"       
      ,"args.camera.far"        
      ,"args.camera.fovy"       
      );

  auto const wavefrontSize      =  vars.getSizeT       ("wavefrontSize"                );
  auto const windowSize         = *vars.get<glm::uvec2>("windowSize"                   );
  auto const minZBits           =  vars.getUint32      ("rssv.param.minZBits"          );
  auto const memoryOptim        =  vars.getInt32       ("rssv.param.memoryOptim"       );
  auto const memoryFactor       =  vars.getInt32       ("rssv.param.memoryFactor"      );
  auto const useBridgePool      =  vars.getInt32       ("rssv.param.useBridgePool"     );
  auto const scaledQuantization =  vars.getInt32       ("rssv.param.scaledQuantization");
  auto const nnear              =  vars.getFloat       ("args.camera.near"             );
  auto const ffar               =  vars.getFloat       ("args.camera.far"              );
  auto const fovy               =  vars.getFloat       ("args.camera.fovy"             );

  vars.reCreate<Config>("rssv.method.config"
      ,wavefrontSize
      ,windowSize.x
      ,windowSize.y
      ,minZBits
      ,memoryOptim
      ,memoryFactor
      ,useBridgePool
      ,nnear
      ,ffar
      ,fovy
      ,scaledQuantization
      );
}
