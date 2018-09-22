#include<CSSV/DrawCapsProgram.h>
#include<CSSV/DrawCapsShaders.h>
#include<SilhouetteShaders.h>

using namespace std;
using namespace ge::gl;

shared_ptr<Program>createDrawCapsProgram(){
  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,cssv::capsVPSrc),
      make_shared<Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        cssv::capsGPSrc));
  return program;
}

