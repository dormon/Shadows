#include <FunctionPrologue.h>
#include <glm/glm.hpp>
#include <geGL/geGL.h>

void createShadowMask(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","windowSize");

  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  vars.reCreate<ge::gl::Texture>("shadowMask" ,(GLenum)GL_TEXTURE_2D,(GLenum)GL_R32F, 1,(GLsizei)windowSize.x,(GLsizei)windowSize.y);
}

