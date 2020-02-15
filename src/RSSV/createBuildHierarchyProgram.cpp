#include <iostream>
#include <sstream>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <RSSV/config.h>
#include <RSSV/createBuildHierarchyProgram.h>
#include <RSSV/buildHierarchyShader.h>
#include <RSSV/reduceShader.h>
#include <RSSV/mortonShader.h>
#include <RSSV/configShader.h>

void rssv::createBuildHierarchyProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"windowSize"
      ,"wavefrontSize"
      ,"args.camera.near"
      ,"args.camera.far"
      ,"args.camera.fovy"
      ,"rssv.method.config"
      ,"rssv.param.usePadding"   
      ,"rssv.param.discardBackfacing"
      ,"rssv.param.mergedBuffers"
      );

  auto const wavefrontSize     =  vars.getSizeT       ("wavefrontSize"               );
  auto const windowSize        = *vars.get<glm::uvec2>("windowSize"                  );
  auto const nnear             =  vars.getFloat       ("args.camera.near"            );
  auto const ffar              =  vars.getFloat       ("args.camera.far"             );
  auto const fovy              =  vars.getFloat       ("args.camera.fovy"            );
  auto const usePadding        =  vars.getUint32      ("rssv.param.usePadding"       );
  auto const discardBackfacing =  vars.getUint32      ("rssv.param.discardBackfacing");
  auto const mergedBuffers     =  vars.getInt32       ("rssv.param.mergedBuffers"    );
  auto const cfg               = *vars.get<Config    >("rssv.method.config"          );

#define PRINT(x) std::cerr << #x ": " << x << std::endl

  PRINT(wavefrontSize);
  PRINT(cfg.minZBits);
  PRINT(nnear);
  PRINT(ffar);
  PRINT(fovy);

  vars.reCreate<ge::gl::Program>("rssv.method.buildHierarchyProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        ballotSrc,
        ge::gl::Shader::define("WARP"               ,(uint32_t)wavefrontSize    ),
        ge::gl::Shader::define("WINDOW_X"           ,(uint32_t)windowSize.x     ),
        ge::gl::Shader::define("WINDOW_Y"           ,(uint32_t)windowSize.y     ),
        ge::gl::Shader::define("MIN_Z_BITS"         ,(uint32_t)cfg.minZBits     ),
        ge::gl::Shader::define("NEAR"               ,nnear                      ),
        glm::isinf(ffar)?ge::gl::Shader::define("FAR_IS_INFINITE"):ge::gl::Shader::define("FAR",ffar),
        ge::gl::Shader::define("FOVY"               ,fovy                       ),
        ge::gl::Shader::define("TILE_X"             ,cfg.tileX                  ),
        ge::gl::Shader::define("TILE_Y"             ,cfg.tileY                  ),
        ge::gl::Shader::define("MEMORY_OPTIM"       ,(int)cfg.memoryOptim       ),
        ge::gl::Shader::define("MEMORY_FACTOR"      ,(uint32_t)cfg.memoryFactor ),
        rssv::configShader,
        rssv::mortonShader,
        rssv::reduceShader,
        ge::gl::Shader::define("USE_PADDING"        ,(int)usePadding            ),
        ge::gl::Shader::define("DISCARD_BACK_FACING",(int)discardBackfacing     ),
        ge::gl::Shader::define("MERGED_BUFFERS"     ,(int)mergedBuffers         ),
        rssv::buildHierarchyShader
        ));
}
