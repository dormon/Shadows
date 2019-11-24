#include <Vars/Vars.h>
#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <SilhouetteShaders.h>

#include <CSSV/caps/createProgram.h>
#include <CSSV/caps/shaders.h>

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
