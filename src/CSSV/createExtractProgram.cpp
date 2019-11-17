#include <CSSV/createExtractProgram.h>
#include <Vars/Vars.h>
#include <FunctionPrologue.h>
#include <geGL/geGL.h>
#include <CSSV/ExtractSilhouetteShader.h>
#include <SilhouetteShaders.h>
#include <FastAdjacency.h>

using namespace ge::gl;
using namespace std;

void cssv::createExtractProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method",
      "cssv.param.alignment"      ,
      "cssv.param.computeSidesWGS",
      "cssv.param.localAtomic"    ,
      "cssv.param.cullSides"      ,
      "cssv.param.usePlanes"      ,
      "cssv.param.useInterleaving",
      "maxMultiplicity"           ,
      "adjacency"                 );

  auto adj = vars.get<Adjacency>("adjacency");
  vars.reCreate<Program>("cssv.method.extractProgram",
      make_shared<Shader>(GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("ALIGN_SIZE"      ,uint32_t(vars.getSizeT ("cssv.param.alignment"      ))),
        Shader::define("WORKGROUP_SIZE_X",int32_t (vars.getUint32("cssv.param.computeSidesWGS"))),
        Shader::define("LOCAL_ATOMIC"    ,int32_t (vars.getBool  ("cssv.param.localAtomic"    ))),
        Shader::define("CULL_SIDES"      ,int32_t (vars.getBool  ("cssv.param.cullSides"      ))),
        Shader::define("USE_PLANES"      ,int32_t (vars.getBool  ("cssv.param.usePlanes"      ))),
        Shader::define("USE_INTERLEAVING",int32_t (vars.getBool  ("cssv.param.useInterleaving"))),
        Shader::define("MAX_MULTIPLICITY",int32_t (vars.getUint32("maxMultiplicity"           ))),
        Shader::define("NOF_EDGES"       ,uint32_t(adj->getNofEdges()                          )),
        silhouetteFunctions,
        computeSrc));
}
