#include<geGL/geGL.h>

#include<FunctionPrologue.h>

#include<CSSV/sides/createDrawProgram.h>
#include<CSSV/sides/drawShaders.h>

using namespace std;
using namespace ge::gl;

void cssv::sides::createDrawProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method");

  vars.reCreate<Program>("cssv.method.sides.drawProgram",
      make_shared<Shader>(GL_VERTEX_SHADER         ,cssv::sides::drawVPSrc),
      make_shared<Shader>(GL_TESS_CONTROL_SHADER   ,cssv::sides::drawCPSrc),
      make_shared<Shader>(GL_TESS_EVALUATION_SHADER,cssv::sides::drawEPSrc));
}

