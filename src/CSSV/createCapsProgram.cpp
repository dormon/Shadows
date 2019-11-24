#include <CSSV/createCapsProgram.h>
#include <Vars/Vars.h>
#include <SilhouetteShaders.h>
#include <CSSV/capsShaders.h>
#include <geGL/geGL.h>
#include <FunctionPrologue.h>

using namespace std;
using namespace ge::gl;

void cssv::createCapsProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method");

  vars.reCreate<Program>(
      "cssv.method.caps.program",
      make_shared<Shader>(GL_VERTEX_SHADER  ,cssv::capsVPSrc),
      make_shared<Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        cssv::capsGPSrc));
}
