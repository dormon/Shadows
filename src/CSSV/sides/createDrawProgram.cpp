#include<geGL/geGL.h>

#include<FunctionPrologue.h>

#include<CSSV/sides/createDrawProgram.h>
#include<CSSV/sides/drawShaders.h>

using namespace std;
using namespace ge::gl;

void cssv::sides::createDrawProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method"
      ,"cssv.param.dontExtractMultiplicity"
      ,"cssv.param.dontPackMult"
      );

  bool const dontExtMult     = vars.getBool("cssv.param.dontExtractMultiplicity");
  bool const dontPackMult    = vars.getBool("cssv.param.dontPackMult");

  if(dontExtMult){
    vars.reCreate<Program>("cssv.method.sides.drawProgram",
        make_shared<Shader>(GL_VERTEX_SHADER         ,"#version 450 core\n",cssv::sides::drawVPSrc),
        make_shared<Shader>(GL_TESS_CONTROL_SHADER   ,cssv::sides::drawCPSrc),
        make_shared<Shader>(GL_TESS_EVALUATION_SHADER,cssv::sides::drawEPSrc));
  }else{
    auto const alignedNofEdges = vars.getUint32("cssv.method.alignedNofEdges");
    vars.reCreate<Program>("cssv.method.sides.drawProgram",
        make_shared<Shader>(GL_VERTEX_SHADER         ,"#version 450 core\n",Shader::define("EXTRACT_MULTIPLICITY",1),cssv::sides::drawVPSrc),
        //make_shared<Shader>(GL_GEOMETRY_SHADER       ,"#version 450 core\n",cssv::sides::drawGPSrc));
        make_shared<Shader>(GL_GEOMETRY_SHADER       ,
            "#version 450 core\n",
            dontPackMult?Shader::define("DONT_PACK_MULT",1):"",
            Shader::define("ALIGNED_NOF_EDGES",alignedNofEdges),
            cssv::sides::drawGPSrc));
  }
}

