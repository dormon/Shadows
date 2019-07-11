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
#include <startStop.h>

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
  size_t const nofPix = gBuffer->depth->getWidth(0)*gBuffer->depth->getHeight(0);
  std::vector<float>data(nofPix);
  //glGetTextureImage(gBuffer->depth->getId(),0,GL_DEPTH_COMPONENT,GL_FLOAT,sizeof(float)*data.size(),data.data());
  glGetTextureImage(pointCloudDepth->getId(),0,GL_DEPTH_COMPONENT,GL_FLOAT,sizeof(float)*data.size(),data.data());

  start(vars,"cpu_computeMinMaxDepth");
  float mmin = 10e10;
  float mmax = -10e10;
  for(auto const&p:data){
    mmin = glm::min(mmin,p);
    mmax = glm::max(mmax,p);
  }
  stop(vars,"cpu_computeMinMaxDepth");


  float near = getCameraNear(vars);
  float far  = getCameraFar(vars);
  auto const depthToZ = [](float d,float near,float far){
    return 2*near*far/(d*(far-near)-far-near);
  };

  auto const depthToZInf = [](float d,float near,float far){
    return 2.f*near / (d - 1.f);
  };

  std::cerr << "near: " << near << " - far: " << far << std::endl;
  std::cerr << "mmin: " << depthToZInf(mmin,near,far) << " - mmax: " << depthToZInf(mmax,near,far) << std::endl;
}

