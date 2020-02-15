#include <iostream>
#include <sstream>

#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <BallotShader.h>

#include <RSSV/config.h>
#include <RSSV/createBuildHierarchyProgram.h>
#include <RSSV/getConfigShader.h>
#include <RSSV/buildHierarchyShader.h>
#include <RSSV/reduceShader.h>
#include <RSSV/mortonShader.h>

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

  auto const usePadding        =  vars.getUint32      ("rssv.param.usePadding"       );
  auto const discardBackfacing =  vars.getUint32      ("rssv.param.discardBackfacing");
  auto const mergedBuffers     =  vars.getInt32       ("rssv.param.mergedBuffers"    );
  auto const cfg               = *vars.get<Config    >("rssv.method.config"          );

#define PRINT(x) std::cerr << #x ": " << x << std::endl

  vars.reCreate<ge::gl::Program>("rssv.method.buildHierarchyProgram",
      std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        ballotSrc,
        getConfigShader(vars),
        rssv::mortonShader,
        rssv::reduceShader,
        ge::gl::Shader::define("USE_PADDING"        ,(int)usePadding            ),
        ge::gl::Shader::define("DISCARD_BACK_FACING",(int)discardBackfacing     ),
        ge::gl::Shader::define("MERGED_BUFFERS"     ,(int)mergedBuffers         ),
        rssv::buildHierarchyShader
        ));
}
