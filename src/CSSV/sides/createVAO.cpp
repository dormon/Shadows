#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <ShadowMethod.h>

#include <CSSV/sides/createVAO.h>

using namespace ge::gl;

void cssv::sides::createVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method","cssv.method.silhouettes");

  auto silhouettes = vars.get<Buffer>("cssv.method.silhouettes");
  auto vao = vars.reCreate<VertexArray>("cssv.method.sides.vao");
  vao->addAttrib(silhouettes,0,componentsPerVertex4D,GL_FLOAT);
}

