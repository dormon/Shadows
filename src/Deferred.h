#pragma once

#include<geGL/geGL.h>

class GBuffer: public ge::gl::Context{
  public:
    GBuffer(uint32_t w,uint32_t h){
      assert(this!=nullptr);
      color       = std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_RGBA16UI,1,w,h);
      position    = std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_RGBA32F ,1,w,h);
      normal      = std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_RGBA32F ,1,w,h);
      triangleIds = std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_R32UI   ,1,w,h);
      depth       = std::make_shared<ge::gl::Texture>(GL_TEXTURE_RECTANGLE,GL_DEPTH24_STENCIL8,1,w,h);
      fbo = std::make_shared<ge::gl::Framebuffer>();
      fbo->attachTexture(GL_COLOR_ATTACHMENT0,color      );
      fbo->attachTexture(GL_COLOR_ATTACHMENT1,position   );
      fbo->attachTexture(GL_COLOR_ATTACHMENT2,normal     );
      fbo->attachTexture(GL_COLOR_ATTACHMENT3,triangleIds);
      fbo->attachTexture(GL_DEPTH_ATTACHMENT ,depth      );
      fbo->drawBuffers({GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3});
      assert(fbo->check());
    }
    void begin(){
      fbo->bind();
    }
    void end(){
      fbo->unbind();
    }
    std::shared_ptr<ge::gl::Texture>position    = nullptr;
    std::shared_ptr<ge::gl::Texture>color       = nullptr;
    std::shared_ptr<ge::gl::Texture>normal      = nullptr;
    std::shared_ptr<ge::gl::Texture>triangleIds = nullptr;
    std::shared_ptr<ge::gl::Texture>depth       = nullptr;
    std::shared_ptr<ge::gl::Framebuffer>fbo     = nullptr;
};
