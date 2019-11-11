#include <Vars/Vars.h>
#include <glm/glm.hpp>
#include<FreeImagePlus.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <Deferred.h>

void takeAScreenShot(vars::Vars&vars){
  fipImage img;
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  //img.setSize(FIT_BITMAP,->x,vars.get<glm::uvec2>("windowSize")->y,24);
  auto id = vars.get<ge::gl::Texture>("shadowMask")->getId();
  std::vector<float>buf(windowSize.x * windowSize.y);
  ge::gl::glGetTextureImage(id,0,GL_RED,GL_FLOAT,(GLsizei)buf.size()*sizeof(float),buf.data());
  img.setSize(FIT_FLOAT,windowSize.x,windowSize.y,32);
  for(size_t y=0;y<windowSize.y;++y){
    auto ptr = (float*)FreeImage_GetScanLine(img,(int32_t)y);
    for(size_t x=0;x<windowSize.x;++x)
      ptr[x] = buf.at(y*windowSize.x + x);
  }
  img.save("/home/dormon/Desktop/test.exr");

  id = vars.get<GBuffer>("gBuffer")->color->getId();
  std::vector<uint8_t>buf1(windowSize.x * windowSize.y * sizeof(uint16_t) * 4);
  ge::gl::glGetTextureImage(id,0,GL_RGBA_INTEGER,GL_UNSIGNED_SHORT, (GLsizei)buf1.size(),buf1.data());
  img.setSize(FIT_BITMAP,windowSize.x,windowSize.y,24);
  for(size_t y=0;y<windowSize.y;++y){
    auto ptr = (uint8_t*)FreeImage_GetScanLine(img,(int32_t)y);
    for(size_t x=0;x<windowSize.x;++x){
      ptr[x*3+0] = buf1.at((y*windowSize.x + x)*(sizeof(uint16_t)*4) + 0*sizeof(uint16_t));
      ptr[x*3+1] = buf1.at((y*windowSize.x + x)*(sizeof(uint16_t)*4) + 1*sizeof(uint16_t));
      ptr[x*3+2] = buf1.at((y*windowSize.x + x)*(sizeof(uint16_t)*4) + 2*sizeof(uint16_t));
    }
  }
  img.save("/home/dormon/Desktop/aa.png");


  
  std::cerr << "take a screenshot" << std::endl;
}
