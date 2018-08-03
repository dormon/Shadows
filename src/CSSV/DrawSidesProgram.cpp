#include<CSSV/DrawSidesProgram.h>

using namespace std;
using namespace ge::gl;

shared_ptr<Program>createDrawSidesProgram(){
#include<CSSV/DrawSidesShaders.h>
  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER         ,drawVPSrc),
      make_shared<Shader>(GL_TESS_CONTROL_SHADER   ,drawCPSrc),
      make_shared<Shader>(GL_TESS_EVALUATION_SHADER,drawEPSrc));
  return program;
}

