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
  vars.reCreate<Texture>("gBufferAsPointCloud",GL_TEXTURE_2D,GL_RGBA32F,1,window->x,window->y);
}

void saveGBufferAsPointCloud(vars::Vars&vars){
  FUNCTION_CALLER();
  createPointCloud(vars);
  auto pointCould = vars.get<Texture>("gBufferAsPointCloud");
  auto gBuffer = vars.get<GBuffer>("gBuffer");
  copyTexture(pointCould,&*gBuffer->position,vars);
}

