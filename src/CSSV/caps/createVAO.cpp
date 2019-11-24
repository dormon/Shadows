#include<Vars/Vars.h>
#include<geGL/geGL.h>

#include<FunctionPrologue.h>

#include<CSSV/caps/createVAO.h>

using namespace std;
using namespace ge::gl;

void cssv::createCapsVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","cssv.method.caps.buffer");

  auto caps = vars.get<Buffer>("cssv.method.caps.buffer");

  auto vao = vars.reCreate<VertexArray>("cssv.method.caps.vao");
  vao->addAttrib(caps,0,4,GL_FLOAT);
}
