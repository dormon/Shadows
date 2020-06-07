#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <RSSV/getConfigShader.h>
#include <RSSV/config.h>
#include <RSSV/createPropagateAABBProgram.h>
#include <RSSV/propagateAABBShader.h>

void rssv::createPropagateAABBProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"windowSize"
      ,"wavefrontSize"
      ,"rssv.method.config"
      );

  auto const nofWarps      =  vars.getUint32      ("rssv.param.propagateWarps");
  auto const cfg           = *vars.get<Config    >("rssv.method.config"       );


  vars.reCreate<ge::gl::Program>("rssv.method.propagateAABBProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        ballotSrc,
        rssv::getConfigShader(vars),

        ge::gl::Shader::define("NOF_WARPS"     ,(uint32_t)nofWarps         ),
        ge::gl::Shader::define("USE_BRIDGE_POOL",(int)cfg.useBridgePool    ),
        rssv::propagateAABBShader
        ));
}
