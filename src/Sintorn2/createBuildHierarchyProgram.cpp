#include <iostream>
#include <sstream>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <Sintorn2/createBuildHierarchyProgram.h>
#include <Sintorn2/buildHierarchyShader.h>
#include <Sintorn2/mortonShader.h>
#include <Sintorn2/quantizeZShader.h>
#include <Sintorn2/depthToZShader.h>
#include <Sintorn2/configShader.h>

void sintorn2::createBuildHierarchyProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2",
      "windowSize",
      "wavefrontSize",
      "args.camera.near",
      "args.camera.far",
      "args.camera.fovy",
      "sintorn2.param.minZBits",
      "sintorn2.param.tileX"   ,
      "sintorn2.param.tileY"   ,
      );

  auto const wavefrontSize       =  vars.getSizeT           ("wavefrontSize"          );
  auto const windowSize          = *vars.get<glm::uvec2>    ("windowSize"             );
  auto const nnear               =  vars.getFloat           ("args.camera.near"       );
  auto const ffar                =  vars.getFloat           ("args.camera.far"        );
  auto const fovy                =  vars.getFloat           ("args.camera.fovy"       );
  auto const minZBits            =  vars.getUint32          ("sintorn2.param.minZBits");
  auto const tileX               =  vars.getUint32          ("sintorn2.param.tileX"   );
  auto const tileY               =  vars.getUint32          ("sintorn2.param.tileY"   );

#define PRINT(x) std::cerr << #x ": " << x << std::endl

  PRINT(wavefrontSize);
  PRINT(minZBits);
  PRINT(nnear);
  PRINT(ffar);
  PRINT(fovy);

  vars.reCreate<ge::gl::Program>("sintorn2.method.buildHierarchyProgram",
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
        sintorn2::configShader,
        sintorn2::mortonShader,
        sintorn2::depthToZShader,
        sintorn2::quantizeZShader,
        sintorn2::reduceShader,
        sintorn2::buildHierarchyShader
        ));
}
