#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <Vars/Vars.h>
#include <util.h>

#include <geGL/OpenGLUtil.h>

std::string getLayoutType(GLenum type){
  if(type == GL_RGBA32F       )return "rgba32f"       ;
  if(type == GL_RGB32F        )return "rgba32f"       ;
  if(type == GL_RG32F         )return "rg32f"         ;
  if(type == GL_R32F          )return "r32f"          ;
  if(type == GL_RGBA16F       )return "rgba16f"       ;
  if(type == GL_RGB16F        )return "rgba16f"       ;
  if(type == GL_RG16F         )return "rg16f"         ;
  if(type == GL_R16F          )return "r16f"          ;
  if(type == GL_R11F_G11F_B10F)return "r11f_g11f_b10f";
  if(type == GL_RGB10_A2      )return "GL_RGB10_A2"   ;
  if(type == GL_RGBA16        )return "rgba16"        ;
  if(type == GL_RGB16         )return "rgba16"        ;
  if(type == GL_RG16          )return "rg16"          ;
  if(type == GL_R16           )return "r16"           ;
  if(type == GL_RGBA8         )return "rgba8"         ;
  if(type == GL_RGB8          )return "rgba8"         ;
  if(type == GL_RG8           )return "rg8"           ;
  if(type == GL_R8            )return "r8"            ;
  if(type == GL_RGBA16_SNORM  )return "rgba16_snorm"  ;
  if(type == GL_RGBA8_SNORM   )return "rgba8_snorm"   ;
  if(type == GL_RG16_SNORM    )return "rg16_snorm"    ;
  if(type == GL_RG8_SNORM     )return "rg8_snorm"     ;
  if(type == GL_R16_SNORM     )return "r16_snorm"     ;
  if(type == GL_R8_SNORM      )return "r8_snorm"      ;

  if(type == GL_RGBA32UI      )return "rgba32ui"      ;
  if(type == GL_RGB32UI       )return "rgba32ui"      ;
  if(type == GL_RG32UI        )return "rg32ui"        ;
  if(type == GL_R32UI         )return "r32ui"         ;
  if(type == GL_RGBA16UI      )return "rgba16ui"      ;
  if(type == GL_RGB16UI       )return "rgba16ui"      ;
  if(type == GL_RG16UI        )return "rg16ui"        ;
  if(type == GL_R16UI         )return "r16ui"         ;
  if(type == GL_RGBA8UI       )return "rgba8ui"       ;
  if(type == GL_RGB8UI        )return "rgba8ui"       ;
  if(type == GL_RG8UI         )return "rg8ui"         ;
  if(type == GL_R8UI          )return "r8ui"          ;
  if(type == GL_RGB10_A2UI    )return "rgb10_a2ui"    ;

  if(type == GL_RGBA32I       )return "rgba32i"       ;
  if(type == GL_RGB32I        )return "rgba32i"       ;
  if(type == GL_RG32I         )return "rg32i"         ;
  if(type == GL_R32I          )return "r32i"          ;
  if(type == GL_RGBA16I       )return "rgba16i"       ;
  if(type == GL_RGB16I        )return "rgba16i"       ;
  if(type == GL_RG16I         )return "rg16i"         ;
  if(type == GL_R16I          )return "r16i"          ;
  if(type == GL_RGBA8I        )return "rgba8i"        ;
  if(type == GL_RGB8I         )return "rgba8i"        ;
  if(type == GL_RG8I          )return "rg8i"          ;
  if(type == GL_R8I           )return "r8i"           ;
  return "";
}

std::string getImageType(GLenum type){
  if(type == GL_RGBA32F       )return "image2D";
  if(type == GL_RGB32F        )return "image2D";
  if(type == GL_RG32F         )return "image2D";
  if(type == GL_R32F          )return "image2D";
  if(type == GL_RGBA16F       )return "image2D";
  if(type == GL_RGB16F        )return "image2D";
  if(type == GL_RG16F         )return "image2D";
  if(type == GL_R16F          )return "image2D";
  if(type == GL_R11F_G11F_B10F)return "image2D";
  if(type == GL_RGB10_A2      )return "image2D";
  if(type == GL_RGBA16        )return "image2D";
  if(type == GL_RGB16         )return "image2D";
  if(type == GL_RG16          )return "image2D";
  if(type == GL_R16           )return "image2D";
  if(type == GL_RGBA8         )return "image2D";
  if(type == GL_RGB8          )return "image2D";
  if(type == GL_RG8           )return "image2D";
  if(type == GL_R8            )return "image2D";
  if(type == GL_RGBA16_SNORM  )return "image2D";
  if(type == GL_RGBA8_SNORM   )return "image2D";
  if(type == GL_RG16_SNORM    )return "image2D";
  if(type == GL_RG8_SNORM     )return "image2D";
  if(type == GL_R16_SNORM     )return "image2D";
  if(type == GL_R8_SNORM      )return "image2D";

  if(type == GL_RGBA32UI      )return "uimage2D";
  if(type == GL_RGB32UI       )return "uimage2D";
  if(type == GL_RG32UI        )return "uimage2D";
  if(type == GL_R32UI         )return "uimage2D";
  if(type == GL_RGBA16UI      )return "uimage2D";
  if(type == GL_RGB16UI       )return "uimage2D";
  if(type == GL_RG16UI        )return "uimage2D";
  if(type == GL_R16UI         )return "uimage2D";
  if(type == GL_RGBA8UI       )return "uimage2D";
  if(type == GL_RGB8UI        )return "uimage2D";
  if(type == GL_RG8UI         )return "uimage2D";
  if(type == GL_R8UI          )return "uimage2D";
  if(type == GL_RGB10_A2UI    )return "uimage2D";

  if(type == GL_RGBA32I       )return "iimage2D";
  if(type == GL_RGB32I        )return "iimage2D";
  if(type == GL_RG32I         )return "iimage2D";
  if(type == GL_R32I          )return "iimage2D";
  if(type == GL_RGBA16I       )return "iimage2D";
  if(type == GL_RGB16I        )return "iimage2D";
  if(type == GL_RG16I         )return "iimage2D";
  if(type == GL_R16I          )return "iimage2D";
  if(type == GL_RGBA8I        )return "iimage2D";
  if(type == GL_RGB8I         )return "iimage2D";
  if(type == GL_RG8I          )return "iimage2D";
  if(type == GL_R8I           )return "iimage2D";
  return "";

}

std::string getProgramName(GLenum type){
  return std::string("copyTexture2D")+getLayoutType(type);
}

void createCopyTexture2DProgram(vars::Vars&vars,GLenum type){
  if(vars.has(getProgramName(type)))return;

  vars.add<ge::gl::Program>(getProgramName(type),std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,R".(
  #version 450
  layout(local_size_x=16,local_size_y=16)in;
  layout(binding=0,)."+getLayoutType(type)+R".()uniform )."+getImageType(type)+R".( outImage;
  layout(binding=1,)."+getLayoutType(type)+R".()uniform )."+getImageType(type)+R".( inImage ;
  uniform uvec2 size = uvec2(512,512);

  void main(){
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(uvec2(coord),size)))return;

    imageStore(outImage,coord,imageLoad(inImage,coord));
  }

  )."));
}

void copyTexture2D(ge::gl::Texture*const out,ge::gl::Texture const*const in,vars::Vars&vars){
  GLenum format = in->getInternalFormat(0);
  createCopyTexture2DProgram(vars,format);
  auto prg = vars.get<ge::gl::Program>(getProgramName(format));
  uint32_t width = in->getWidth(0);
  uint32_t height = in->getHeight(0);
  in->bindImage(1);
  out->bindImage(0);
  prg
    ->set2ui("size",width,height)
    ->dispatch(divRoundUp(width,16),divRoundUp(height,16),1);
  ge::gl::glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void blitDepthStencil(ge::gl::Texture*const out,ge::gl::Texture*const in){
  std::cerr << "blitDepthStencil" << std::endl;
  std::cerr << "error: " << ge::gl::glGetError() << std::endl;
  GLuint fbos[2];
  GLint w = in->getWidth(0);
  GLint h = in->getHeight(0);
  std::cerr << "w h : " << w << " " << h << std::endl;
  ge::gl::glCreateFramebuffers(2,fbos);
  ge::gl::glNamedFramebufferTexture(fbos[0],GL_DEPTH_ATTACHMENT,out->getId(),0);
  ge::gl::glNamedFramebufferTexture(fbos[1],GL_DEPTH_ATTACHMENT,in ->getId(),0);
  ge::gl::glBlitNamedFramebuffer(fbos[1],fbos[0],0,0,w,h,0,0,w,h,
       GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT,GL_NEAREST);
  ge::gl::glDeleteFramebuffers(2,fbos);

  //auto read  = std::make_shared<ge::gl::Framebuffer>();
  //auto write = std::make_shared<ge::gl::Framebuffer>();
  //read->attachTexture(GL_DEPTH_ATTACHMENT,in);
  //read->attachTexture(GL_STENCIL_ATTACHMENT,in);
  //write->attachTexture(GL_DEPTH_ATTACHMENT,out);
  //write->attachTexture(GL_STENCIL_ATTACHMENT,out);
  //GLint w = in->getWidth(0);
  //GLint h = in->getHeight(0);
  //std::cerr << "w x h: " << w << " x " << h << std::endl;
  //ge::gl::glBlitNamedFramebuffer(read->getId(),write->getId(),0,0,w,h,0,0,w,h,
  //    GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT,GL_NEAREST);
  //ge::gl::glFinish();
  std::cerr << "error: " << ge::gl::glGetError() << std::endl;
}

void copyTextureRectangle(ge::gl::Texture*const out,ge::gl::Texture*const in,vars::Vars&vars){
  GLenum format = in->getInternalFormat(0);
  if(format == GL_DEPTH24_STENCIL8)blitDepthStencil(out,in);
}

void copyTexture(ge::gl::Texture*const out,ge::gl::Texture*const in,vars::Vars&vars){
  if(in->getInternalFormat(0) != out->getInternalFormat(0))return;
  if(in->getWidth(0) != out->getWidth(0))return;
  if(in->getHeight(0) != out->getHeight(0))return;
  if(in->getDepth(0) != out->getDepth(0))return;
  if(in->getTarget() != out->getTarget())return;
  if(in->getTarget() == GL_TEXTURE_2D)copyTexture2D(out,in,vars);
  if(in->getTarget() == GL_TEXTURE_RECTANGLE)copyTextureRectangle(out,in,vars);
}
