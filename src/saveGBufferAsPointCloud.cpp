#include <Vars/Vars.h>
#include <geGL/Texture.h>
#include <geGL/StaticCalls.h>
#include <glm/glm.hpp>
#include <Deferred.h>
#include <FunctionPrologue.h>
#include <copyTexture.h>
#include <getMVP.h>
#include <getCameraNear.h>
#include <getCameraFar.h>

using namespace ge::gl;
using namespace glm;

void createPointCloud(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","windowSize");

  auto window = vars.get<uvec2>("windowSize");
  uint32_t w = window->x;
  uint32_t h = window->y;
               vars.reCreate<Texture>("pointCloud.positionTexture",GL_TEXTURE_2D       ,GL_RGBA32F         ,1,w,h);
  auto color = vars.reCreate<Texture>("pointCloud.colorTexture"   ,GL_TEXTURE_2D       ,GL_RGBA16UI        ,1,w,h);
               vars.reCreate<Texture>("pointCloud.depthTexture"   ,GL_TEXTURE_RECTANGLE,GL_DEPTH24_STENCIL8,1,w,h);
  color->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
  color->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST);
}

void saveGBufferAsPointCloud(vars::Vars&vars){
  FUNCTION_CALLER();
  createPointCloud(vars);
  vars.reCreate<mat4>("pointCloud.mvp",getMVP(vars));
  vars.reCreate<float>("pointCloud.near",getCameraNear(vars));
  vars.reCreate<float>("pointCloud.far" ,getCameraFar (vars));
  auto pointCloud      = vars.get<Texture>("pointCloud.positionTexture");
  auto pointCloudColor = vars.get<Texture>("pointCloud.colorTexture"   );
  auto pointCloudDepth = vars.get<Texture>("pointCloud.depthTexture"   );
  auto gBuffer = vars.get<GBuffer>("gBuffer");
  copyTexture(pointCloud     ,&*gBuffer->position,vars);
  copyTexture(pointCloudColor,&*gBuffer->color   ,vars);
  copyTexture(pointCloudDepth,&*gBuffer->depth   ,vars);
}

