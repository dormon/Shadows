#include <computeDepthMinMax.h>

#include<geGL/StaticCalls.h>
#include<Vars/Vars.h>
#include<glm/glm.hpp>
#include<divRoundUp.h>
#include<GLSLLine.h>
#include<fillValues.h>

using namespace ge::gl;
using namespace glm;
using namespace std;

class ComputeDepthMinMaxImpl{
  public:
    std::shared_ptr<Program>prg;
    std::vector<std::shared_ptr<Buffer>>levels;
    std::vector<uvec2>sizes;
    size_t width  = 0;
    size_t height = 0;
    size_t tileX  = 0;
    size_t tileY  = 0;
    void initProgram();
    vector<uvec2>computeSizes(size_t w,size_t h,size_t tx,size_t ty)const;
    void resize(size_t w,size_t h,size_t tx,size_t ty);
};


vector<uvec2>ComputeDepthMinMaxImpl::computeSizes(size_t w,size_t h,size_t tx,size_t ty)const{
  vector<uvec2>sizes;
  sizes.push_back(uvec2(w,h));
  while(sizes.back().x != 1 || sizes.back().y != 1){
    auto const&last = sizes.back();
    auto ne = uvec2(divRoundUp(last.x,tx),divRoundUp(last.y,ty));
    sizes.push_back(ne);
  }
  sizes.pop_back(); //remove 1x1
  std::reverse(sizes.begin(),sizes.end());
  sizes.pop_back(); //remove WxH
  return sizes;
}


void ComputeDepthMinMaxImpl::resize(size_t w,size_t h,size_t tx,size_t ty){
  if(w == width && h == height && tileX == tx && tileY == ty)
    return;
  width  = w;
  height = h;
  tileX  = tx;
  tileY  = ty;
  sizes = computeSizes(w,h,tx,ty);
  levels.clear();
  for(auto const&s:sizes)
    levels.push_back(make_shared<Buffer>(sizeof(float)*s.x*s.y));
}

void ComputeDepthMinMaxImpl::initProgram(){
  std::string static const src = 
  GLSL_LINE+
  fillValues(R".(
  #version 450

  #define TILE_X %%
  #define TILE_Y %%
  #define WGS (TILE_X*TILE_Y)
  #define WAVEFRONT_SIZE 64

  layout(local_size_x=TILE_X,local_size_Y=TILE_Y)in;

  layout(binding=0)readonly  uniform sampler2DRect inTex;
  layout(binding=1)          uniform sampler2DRect outTex;
  layout(binding=0)writeonly         buffer        outBuffer(float minDepth,float maxDepth);

  shared float local[TILE_X*TILE_Y];

  void loadToLocal(){
    local[gl_LocalInvocationIndex] = texelFetch(inTex,gl_GlobalInvocationID.xy,0).x;
    barrier();
  }

  void firstRun(){
    if(gl_LocalInvocationIndex >= (WGS>>1))return;
    float minMax[2];
    minMax[0] = local[gl_LocalInvocationIndex              ];
    minMax[1] = local[gl_LocalInvocationIndex+activeThreads];
    uint firstLesser = uint(minMax[0] < minMax[1]);
    local[gl_LocalInvocationIndex              ] = minMax[1-firstLesser];
    local[gl_LocalInvocationIndex+activeThreads] = minMax[  firstLesser];
  }

  void reduce(){

  }

  void main(){
    ivec2 size = textureSize(inTex,0);
    if(any(greaterThanEqual(gl_GlobalInvocationID.xy,size)))return;

    loadToLocal();

    firsrRun();


  }

  ).","%%",tileX,tileY);
  prg = make_shared<Program>(make_shared<Shader>(GL_COMPUTE_SHADER,src));

}


void ComputeDepthMinMax::operator()(
    ge::gl::Buffer       *const buffer  ,
    ge::gl::Texture const*const depthTex,
    size_t                      tileX   ,
    size_t                      tileY   ){
  auto const width = depthTex->getWidth(0);
  auto const height = depthTex->getHeight(0);
  impl->resize(width,height,tileX,tileY);
}

