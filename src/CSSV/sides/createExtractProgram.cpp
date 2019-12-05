#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <SilhouetteShaders.h>
#include <FastAdjacency.h>

#include <CSSV/sides/extractShader.h>
#include <CSSV/sides/createExtractProgram.h>

using namespace ge::gl;
using namespace std;

void cssv::sides::createExtractProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method"         ,
      "cssv.param.alignment"              ,
      "cssv.param.computeSidesWGS"        ,
      "cssv.param.noLocalAtomic"          ,
      "cssv.param.cullSides"              ,
      "cssv.param.dontUsePlanes"          ,
      "cssv.param.dontUseInterleaving"    ,
      "cssv.param.dontExtractMultiplicity",
      "cssv.param.dontPackMult"           ,
      "maxMultiplicity"                   ,
      "wavefrontSize"                     ,
      "adjacency"                         );

  auto adj = vars.get<Adjacency>("adjacency");
  vars.reCreate<Program>("cssv.method.extractProgram",
      make_shared<Shader>(GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("WARP"                     ,uint32_t( vars.getSizeT ("wavefrontSize"                     ))),
        Shader::define("ALIGN_SIZE"               ,uint32_t( vars.getSizeT ("cssv.param.alignment"              ))),
        Shader::define("WORKGROUP_SIZE_X"         ,int32_t ( vars.getUint32("cssv.param.computeSidesWGS"        ))),
        Shader::define("LOCAL_ATOMIC"             ,int32_t (!vars.getBool  ("cssv.param.noLocalAtomic"          ))),
        Shader::define("CULL_SIDES"               ,int32_t ( vars.getBool  ("cssv.param.cullSides"              ))),
        Shader::define("USE_PLANES"               ,int32_t (!vars.getBool  ("cssv.param.dontUsePlanes"          ))),
        Shader::define("USE_INTERLEAVING"         ,int32_t (!vars.getBool  ("cssv.param.dontUseInterleaving"    ))),
        Shader::define("DONT_EXTRACT_MULTIPLICITY",int32_t ( vars.getBool  ("cssv.param.dontExtractMultiplicity"))),
        Shader::define("DONT_PACK_MULT"           ,int32_t ( vars.getBool  ("cssv.param.dontPackMult"           ))),
        Shader::define("MAX_MULTIPLICITY"         ,int32_t ( vars.getUint32("maxMultiplicity"                   ))),
        Shader::define("NOF_EDGES"                ,uint32_t( adj->getNofEdges()                                  )),
        silhouetteFunctions,
        computeSrc));
}
