#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <BallotShader.h>
#include <Deferred.h>

#include <RSSV/merge.h>
#include <RSSV/mergeShader.h>
#include <RSSV/configShader.h>
#include <RSSV/mortonShader.h>
#include <RSSV/quantizeZShader.h>
#include <RSSV/depthToZShader.h>
#include <RSSV/config.h>

#include <iomanip>
#include <Timer.h>
#include <bitset>

using namespace ge::gl;
using namespace std;

namespace rssv{
void createMergeProgram(vars::Vars&vars){
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

  vars.reCreate<ge::gl::Program>("rssv.method.mergeProgram",
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
        rssv::configShader,
        rssv::mortonShader,
        rssv::depthToZShader,
        rssv::quantizeZShader,
        rssv::mergeShader
        ));

}

}


void rssv::merge(vars::Vars&vars){
  FUNCTION_CALLER();
  createMergeProgram(vars);

  auto depth            =  vars.get<GBuffer>("gBuffer")->depth;
  auto prg              =  vars.get<Program>("rssv.method.mergeProgram");
  auto nodePool         =  vars.get<Buffer >("rssv.method.nodePool");
  auto shadowMask       =  vars.get<Texture>("shadowMask");

  auto cfg              = *vars.get<Config >("rssv.method.config");

  nodePool        ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  depth->bind(0);
  shadowMask->bindImage(1);
  
  prg->use();
  glDispatchCompute(cfg.clustersX,cfg.clustersY,1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);

}
