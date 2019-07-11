#pragma once

#include<Vars/Fwd.h>
#include<geGL/geGL.h>

class ComputeDepthMinMaxImpl;
class ComputeDepthMinMax{
  public:
    void operator()(
        ge::gl::Buffer       *const buffer       ,
        ge::gl::Texture const*const depthTex     ,
        size_t                      tileX    = 16,
        size_t                      tileY    = 16);
  protected:
    std::unique_ptr<ComputeDepthMinMaxImpl>impl;
};

void computeDepthMinMax(
    ge::gl::Buffer       *const buffer       ,
    ge::gl::Texture const*const depthTex     ,
    vars::Vars                 &vars         ,
    size_t                      tileX    = 16,
    size_t                      tileY    = 16,
    std::string           const&ctxName  = "");
