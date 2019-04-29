#include<geGL/Texture.h>
#include<geGL/OpenGLUtil.h>

size_t basicTexelSize(ge::gl::BasicInternalFormatElement const&info){
  size_t result = 0;
  for(auto const&x:info.channelSize)
    result += x;
  return result;
}

void saveBasicTexture(std::string const&name,ge::gl::Texture const*tex){
  auto const iFormat   = tex->getInternalFormat(0);
  auto const info      = ge::gl::getBasicInternalFormatInformation(iFormat);
  auto const width     = tex->getWidth(0);
  auto const height    = tex->getHeight(0);
  auto const texelSize = basicTexelSize(info);
  auto const nofTexels = width * height;
  auto buffer = std::vector<uint8_t>(texelSize * nofTexels);
  if(info.type == ge::gl::BasicInternalFormatElement::UNSIGNED_INT){
    //ge::gl::glGetTextureImage(); tex,level,format,type,bufSize,pixels
    //
    //ge::gl::glGetTextureSubImage(); tex,level,xoff,yoff,zoff,w,h,d,format,type,bufSize,pixels
  }
}

void saveDepthTexture(std::string const&name,ge::gl::Texture const*tex){
}

void saveCompressedTexture(std::string const&name,ge::gl::Texture const*tex){

}

void saveTexture(std::string const&name,ge::gl::Texture const*tex){
  auto const iFormat   = tex->getInternalFormat(0);
  if(ge::gl::isInternalFormatBasic(iFormat)){
    saveBasicTexture(name,tex);
    return;
  }
  if(ge::gl::isInternalFormatDepth(iFormat)){
    saveDepthTexture(name,tex);
    return;
  }
  if(ge::gl::isInternalFormatDepth(iFormat)){
    saveCompressedTexture(name,tex);
    return;
  }
/*

  auto const info      = ge::gl::getBasicInternalFormatInformation(iFormat);
  auto const nofChannels = ge::gl::nofInternalFormatChannels(iFormat);
  size_t const channelSizes[4] = {
    ge::gl::internalFormatChannelSize(iFormat,0),
    ge::gl::internalFormatChannelSize(iFormat,1),
    ge::gl::internalFormatChannelSize(iFormat,2),
    ge::gl::internalFormatChannelSize(iFormat,3),
  };
  auto const floatingPoint = ge::gl::internalFormatFloatingPoint(iFormat);
  auto const signedType    = ge::gl::internalFormatSigned(iFormat);
  auto const fixedPoint    = ge::gl::internalFormatFixedPoint(iFormat);
  auto const texelSize     = ge::gl::internalFormatSize(iFormat);
  fipImage img;
  std::vector<uint8_t>buf(nofTexels * texelSize);

  ge::gl::glGetTextureImage(id,0,GL_RED,GL_FLOAT,buf.size(),buf.data());
  img.setSize(FIT_FLOAT,width,height,32);
  for(size_t y=0;y<height;++y){
    auto ptr = (float*)FreeImage_GetScanLine(img,y);
    for(size_t x=0;x<width;++x)
      ptr[x] = buf.at(y*width + x);
  }
  img.save(name.c_str());
  */
}
