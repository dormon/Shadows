#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <RSSV/config.h>
#include <RSSV/createPropagateAABBProgram.h>
#include <RSSV/propagateAABBShader.h>
#include <RSSV/configShader.h>

void rssv::createPropagateAABBProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"windowSize"
      ,"wavefrontSize"
      ,"rssv.method.config"
      ,"rssv.param.mergedBuffers"
      );

  auto const wavefrontSize =  vars.getSizeT       ("wavefrontSize"            );
  auto const windowSize    = *vars.get<glm::uvec2>("windowSize"               );
  auto const nofWarps      =  vars.getUint32      ("rssv.param.propagateWarps");
  auto const mergedBuffers =  vars.getInt32       ("rssv.param.mergedBuffers" );
  auto const cfg           = *vars.get<Config    >("rssv.method.config"       );


  vars.reCreate<ge::gl::Program>("rssv.method.propagateAABBProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        ballotSrc,
        ge::gl::Shader::define("WARP"          ,(uint32_t)wavefrontSize    ),
        ge::gl::Shader::define("WINDOW_X"      ,(uint32_t)windowSize.x     ),
        ge::gl::Shader::define("WINDOW_Y"      ,(uint32_t)windowSize.y     ),
        ge::gl::Shader::define("MIN_Z_BITS"    ,(uint32_t)cfg.minZBits     ),
        ge::gl::Shader::define("TILE_X"        ,cfg.tileX                  ),
        ge::gl::Shader::define("TILE_Y"        ,cfg.tileY                  ),
        ge::gl::Shader::define("MEMORY_OPTIM"  ,(int)cfg.memoryOptim       ),
        ge::gl::Shader::define("MEMORY_FACTOR" ,(uint32_t)cfg.memoryFactor ),
        rssv::configShader,
        ge::gl::Shader::define("NOF_WARPS"     ,(uint32_t)nofWarps         ),
        ge::gl::Shader::define("MERGED_BUFFERS",(int)mergedBuffers         ),
        rssv::propagateAABBShader
        ));
}
