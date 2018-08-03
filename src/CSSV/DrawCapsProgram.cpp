#include<CSSV/DrawCapsProgram.h>

using namespace std;
using namespace ge::gl;

shared_ptr<Program>createDrawCapsProgram(){
#include<CSSV/DrawCapsShaders.h>
#include<SilhouetteShaders.h>
  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,capsVPSrc),
      make_shared<Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        capsGPSrc));
  return program;
}

