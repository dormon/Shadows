#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <BallotShader.h>
#include <Deferred.h>

#include <Sintorn2/merge.h>
#include <Sintorn2/mergeShader.h>
#include <Sintorn2/configShader.h>
#include <Sintorn2/mortonShader.h>
#include <Sintorn2/quantizeZShader.h>
#include <Sintorn2/depthToZShader.h>
#include <Sintorn2/config.h>

#include <iomanip>
#include <Timer.h>
#include <bitset>

using namespace ge::gl;
using namespace std;

namespace sintorn2{
void createMergeProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method"
      ,"windowSize"
      ,"wavefrontSize"
      ,"args.camera.near"
      ,"args.camera.far"
      ,"args.camera.fovy"
      ,"sintorn2.param.minZBits"
      ,"sintorn2.param.tileX"   
      ,"sintorn2.param.tileY"   
      );


  auto const wavefrontSize       =  vars.getSizeT           ("wavefrontSize"          );
  auto const windowSize          = *vars.get<glm::uvec2>    ("windowSize"             );
  auto const nnear               =  vars.getFloat           ("args.camera.near"       );
  auto const ffar                =  vars.getFloat           ("args.camera.far"        );
  auto const fovy                =  vars.getFloat           ("args.camera.fovy"       );
  auto const minZBits            =  vars.getUint32          ("sintorn2.param.minZBits");
  auto const tileX               =  vars.getUint32          ("sintorn2.param.tileX"   );
  auto const tileY               =  vars.getUint32          ("sintorn2.param.tileY"   );

  vars.reCreate<ge::gl::Program>("sintorn2.method.mergeProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
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
        sintorn2::mergeShader
        ));

}

}


void sintorn2::merge(vars::Vars&vars){
  FUNCTION_CALLER();
  createMergeProgram(vars);

  auto depth            =  vars.get<GBuffer>("gBuffer")->depth;
  auto prg              =  vars.get<Program>("sintorn2.method.mergeProgram");
  auto nodePool         =  vars.get<Buffer >("sintorn2.method.nodePool");
  auto shadowMask       =  vars.get<Texture>("shadowMask");

  auto cfg              = *vars.get<Config >("sintorn2.method.config");

  nodePool        ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  depth->bind(0);
  shadowMask->bindImage(1);
  
  prg->use();
  glDispatchCompute(cfg.clustersX,cfg.clustersY,1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);

}
