#include <Vars/Vars.h>
#include <geGL/Texture.h>
#include <geGL/StaticCalls.h>
#include <glm/glm.hpp>
#include <Deferred.h>
#include <FunctionPrologue.h>
#include <copyTexture.h>

using namespace ge::gl;
using namespace glm;

void createPointCloud(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","windowSize");
  auto window = vars.get<uvec2>("windowSize");
               vars.reCreate<Texture>("pointCloud.positionTexture",GL_TEXTURE_2D,GL_RGBA32F ,1,window->x,window->y);
  auto color = vars.reCreate<Texture>("pointCloud.colorTexture"   ,GL_TEXTURE_2D,GL_RGBA16UI,1,window->x,window->y);
  color->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
  color->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST);
}

void saveGBufferAsPointCloud(vars::Vars&vars){
  FUNCTION_CALLER();
  createPointCloud(vars);
  auto pointCould      = vars.get<Texture>("pointCloud.positionTexture");
  auto pointCouldColor = vars.get<Texture>("pointCloud.colorTexture"   );
  auto gBuffer = vars.get<GBuffer>("gBuffer");
  copyTexture(pointCould     ,&*gBuffer->position,vars);
  copyTexture(pointCouldColor,&*gBuffer->color   ,vars);
}

