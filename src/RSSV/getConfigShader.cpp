#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <RSSV/config.h>
#include <RSSV/getConfigShader.h>
#include <RSSV/configShader.h>

#include <sstream>

namespace rssv{

std::string getConfigShader(vars::Vars&vars,bool debug = false){
  std::stringstream ss;

  auto const wavefrontSize =  vars.getSizeT       ("wavefrontSize"            );

  auto const cfg = debug?
    *vars.get<Config>("rssv.method.debug.dump.config"):
    *vars.get<Config>("rssv.method.config"           );

  ss << ge::gl::Shader::define("WARP"          ,(uint32_t)wavefrontSize   );
  ss << ge::gl::Shader::define("WINDOW_X"      ,(uint32_t)cfg.windowX     );
  ss << ge::gl::Shader::define("WINDOW_Y"      ,(uint32_t)cfg.windowY     );
  ss << ge::gl::Shader::define("MIN_Z_BITS"    ,(uint32_t)cfg.minZBits    );
  ss << ge::gl::Shader::define("TILE_X"        ,(uint32_t)cfg.tileX       );
  ss << ge::gl::Shader::define("TILE_Y"        ,(uint32_t)cfg.tileY       );
  ss << ge::gl::Shader::define("MEMORY_OPTIM"  ,(int     )cfg.memoryOptim );
  ss << ge::gl::Shader::define("MEMORY_FACTOR" ,(uint32_t)cfg.memoryFactor);
  ss << ge::gl::Shader::define("NEAR"          ,(float   )cfg.nnear       );
  ss << ge::gl::Shader::define("FAR"           ,(float   )cfg.ffar        );
  ss << ge::gl::Shader::define("FOVY"          ,(float   )cfg.fovy        );

  if(glm::isinf(cfg.ffar))
    ss << ge::gl::Shader::define("FAR_IS_INFINITE");

  ss << rssv::configShader;

  return ss.str();
}

std::string getConfigShader(vars::Vars&vars){
  return getConfigShader(vars,false);
}

std::string getDebugConfigShader(vars::Vars&vars){
  return getConfigShader(vars,true);
}

}
