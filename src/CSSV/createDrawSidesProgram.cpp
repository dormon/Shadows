#include<CSSV/createDrawSidesProgram.h>
#include<CSSV/sidesShaders.h>
#include<FunctionPrologue.h>
#include<geGL/geGL.h>

using namespace std;
using namespace ge::gl;

void cssv::createDrawSidesProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method");

  vars.reCreate<Program>("cssv.method.sides.drawProgram",
      make_shared<Shader>(GL_VERTEX_SHADER         ,cssv::drawVPSrc),
      make_shared<Shader>(GL_TESS_CONTROL_SHADER   ,cssv::drawCPSrc),
      make_shared<Shader>(GL_TESS_EVALUATION_SHADER,cssv::drawEPSrc));
}

