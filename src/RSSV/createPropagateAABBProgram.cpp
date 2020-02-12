#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <RSSV/createPropagateAABBProgram.h>
#include <RSSV/propagateAABBShader.h>
#include <RSSV/configShader.h>

void rssv::createPropagateAABBProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"windowSize"
      ,"wavefrontSize"
      ,"rssv.param.minZBits"
      ,"rssv.param.tileX"   
      ,"rssv.param.tileY"   
      ,"rssv.param.memoryOptim"
      );

  auto const wavefrontSize =  vars.getSizeT       ("wavefrontSize"            );
  auto const windowSize    = *vars.get<glm::uvec2>("windowSize"               );
  auto const minZBits      =  vars.getUint32      ("rssv.param.minZBits"      );
  auto const tileX         =  vars.getUint32      ("rssv.param.tileX"         );
  auto const tileY         =  vars.getUint32      ("rssv.param.tileY"         );
  auto const nofWarps      =  vars.getUint32      ("rssv.param.propagateWarps");
  auto const memoryOptim   =  vars.getInt32       ("rssv.param.memoryOptim"   );


  vars.reCreate<ge::gl::Program>("rssv.method.propagateAABBProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        ge::gl::Shader::define("WARP"        ,(uint32_t)wavefrontSize),
        ge::gl::Shader::define("WINDOW_X"    ,(uint32_t)windowSize.x ),
        ge::gl::Shader::define("WINDOW_Y"    ,(uint32_t)windowSize.y ),
        ge::gl::Shader::define("MIN_Z_BITS"  ,(uint32_t)minZBits     ),
        ge::gl::Shader::define("TILE_X"      ,tileX                  ),
        ge::gl::Shader::define("TILE_Y"      ,tileY                  ),
        ge::gl::Shader::define("NOF_WARPS"   ,(uint32_t)nofWarps     ),
        ge::gl::Shader::define("MEMORY_OPTIM",(int)memoryOptim       ),
        ballotSrc,
        rssv::configShader,
        rssv::propagateAABBShader
        ));
}
