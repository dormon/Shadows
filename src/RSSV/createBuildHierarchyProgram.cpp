#include <iostream>
#include <sstream>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <RSSV/createBuildHierarchyProgram.h>
#include <RSSV/buildHierarchyShader.h>
#include <RSSV/mortonShader.h>
#include <RSSV/configShader.h>

void rssv::createBuildHierarchyProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"windowSize"
      ,"wavefrontSize"
      ,"args.camera.near"
      ,"args.camera.far"
      ,"args.camera.fovy"
      ,"rssv.param.minZBits"
      ,"rssv.param.tileX"   
      ,"rssv.param.tileY"   
      );

  auto const wavefrontSize       =  vars.getSizeT           ("wavefrontSize"          );
  auto const windowSize          = *vars.get<glm::uvec2>    ("windowSize"             );
  auto const nnear               =  vars.getFloat           ("args.camera.near"       );
  auto const ffar                =  vars.getFloat           ("args.camera.far"        );
  auto const fovy                =  vars.getFloat           ("args.camera.fovy"       );
  auto const minZBits            =  vars.getUint32          ("rssv.param.minZBits");
  auto const tileX               =  vars.getUint32          ("rssv.param.tileX"   );
  auto const tileY               =  vars.getUint32          ("rssv.param.tileY"   );

#define PRINT(x) std::cerr << #x ": " << x << std::endl

  PRINT(wavefrontSize);
  PRINT(minZBits);
  PRINT(nnear);
  PRINT(ffar);
  PRINT(fovy);

  vars.reCreate<ge::gl::Program>("rssv.method.buildHierarchyProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        //"#extension GL_NV_shader_thread_group : enable\n",
        ge::gl::Shader::define("WARP"      ,(uint32_t)wavefrontSize),
        ge::gl::Shader::define("WINDOW_X"  ,(uint32_t)windowSize.x ),
        ge::gl::Shader::define("WINDOW_Y"  ,(uint32_t)windowSize.y ),
        ge::gl::Shader::define("MIN_Z_BITS",(uint32_t)minZBits     ),
        ge::gl::Shader::define("NEAR"      ,nnear                  ),
        glm::isinf(ffar)?ge::gl::Shader::define("FAR_IS_INFINITE"):ge::gl::Shader::define("FAR",ffar),
        ge::gl::Shader::define("FOVY"      ,fovy                   ),
        ge::gl::Shader::define("TILE_X"    ,tileX                  ),
        ge::gl::Shader::define("TILE_Y"    ,tileY                  ),
        ballotSrc,
        rssv::configShader,
        rssv::mortonShader,
        rssv::reduceShader,
        rssv::buildHierarchyShader
        ));
}
