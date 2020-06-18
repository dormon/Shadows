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
#include <RSSV/mergeMainShader.h>
#include <RSSV/getConfigShader.h>

#include <iomanip>
#include <Timer.h>
#include <bitset>

using namespace ge::gl;
using namespace std;

namespace rssv{
void createMergeProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"wavefrontSize"
      ,"rssv.method.config"
      ,"rssv.param.performMerge"
      );

  auto const performMerge                 =  vars.getBool        ("rssv.param.performMerge"                );
  vars.reCreate<ge::gl::Program>("rssv.method.mergeProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER
        ,"#version 450\n"
        ,ballotSrc
        ,getConfigShader(vars)
        ,Shader::define("PERFORM_MERGE"                 ,(int     )performMerge                )
        ,rssv::demortonShader
        ,rssv::mortonShader
        ,rssv::depthToZShader
        ,rssv::quantizeZShader
        ,rssv::mergeShaderFWD
        ,rssv::mergeMainShader
        ,rssv::mergeShader
        ));
}

}


void rssv::merge(vars::Vars&vars){
  FUNCTION_CALLER();

  auto const mergeInMega = vars.getBool("rssv.param.mergeInMega");
  if(mergeInMega)return;

  createMergeProgram(vars);

  auto depth      = vars.get<GBuffer>("gBuffer")->depth;
  auto shadowMask = vars.get<Texture>("shadowMask");
  auto stencil    = vars.get<Texture>("rssv.method.stencil");

  auto prg        =  vars.get<Program>("rssv.method.mergeProgram");

  depth     ->bind     (0);
  shadowMask->bindImage(1);
  stencil   ->bindImage(2);
  auto bridges = vars.get<Buffer >("rssv.method.bridges"                   );
  prg->bindBuffer("Bridges",bridges);
  
  prg->use();
  glDispatchCompute(1000,1,1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);

}
