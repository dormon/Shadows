#include<StencilBufferToShadowMaskProgram.h>

using namespace std;
using namespace ge::gl;

shared_ptr<Program>createStencilBufferToShadowMaskProgram(){
#include<ShadowVolumesShaders.h>
  auto program = std::make_shared<ge::gl::Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,convertStencilBufferToShadowMaskVPSrc),
      make_shared<Shader>(GL_FRAGMENT_SHADER,convertStencilBufferToShadowMaskFPSrc));
  return program;
}
