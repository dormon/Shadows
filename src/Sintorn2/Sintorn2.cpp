#include <Sintorn2/Sintorn2.h>
#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>
#include <startStop.h>

Sintorn2::Sintorn2(vars::Vars& vars) : ShadowMethod(vars) {}

Sintorn2::~Sintorn2() {}

void allocateHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2","windowSize","wavefrontSize");

  auto wavefrontSize =  vars.getSizeT("wavefrontSize");
  auto windowSize    = *vars.get<glm::uvec2>("windowSize");
  if(wavefrontSize != 64)throw std::runtime_error("Sintorn2::allocateHierarchy - only 64 warp size supported");

}

void createProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2","windowSize","wavefrontSize");
  auto const src = R".(
  layout(local_size_x=16,local_size_y=16)in;

  layout(binding=2)uniform sampler2DRect depthTexture;
  uniform uvec2 windowSize = uvec2(512,512);

  uniform float near = 0.01f;
  uniform uint Sy = 512/8;
  uniform float fovy = 90;

  uint convertDepth(float depth){
    return uint(log(-depth/near) / log(1+2*tan(fovy)/Sy));
  }

  uint morton(uvec3 cc){
    uint res = 0;
    res |= (cc.y& 1)<< 0;//0
    res |= (cc.x& 1)<< 1;//1
    res |= (cc.z& 1)<< 2;//2
    res |= (cc.y& 2)<< 2;//3
    res |= (cc.x& 2)<< 3;//4
    res |= (cc.z& 2)<< 4;//5
    res |= (cc.y& 4)<< 4;//6
    res |= (cc.x& 4)<< 5;//7
    res |= (cc.z& 4)<< 6;//8
    res |= (cc.y& 8)<< 6;//9
    res |= (cc.x& 8)<< 7;//10
    res |= (cc.z& 8)<< 8;//11
    res |= (cc.y&16)<< 8;//12
    res |= (cc.x&16)<< 9;//13
    res |= (cc.z&16)<<10;//14
    res |= (cc.y&32)<<10;//15
    res |= (cc.x&32)<<11;//16
    res |= (cc.z&32)<<12;//17
    res |= (cc.y&64)<<12;//18
    res |= (cc.x&64)<<13;//19
    res |= (cc.z&64)<<14;//20a
  }

  #define GID uvec2(gl_GlobalInvocationID.xy)
  void main(){
    if(any(greaterThanEqual(GID,windowSize)))return;

    float depth = texelFetch(depthTexture,ivec2(GID),0).x*2-1;
    uvec3 clusterCoord = uvec3(uvec2(GID/8) , convertDepth(depth));
    
  }


  ).";

}

void Sintorn2::create(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  auto depth = vars.get<GBuffer>("gBuffer")->depth;
  auto width = depth->getWidth(0);
  auto height = depth->getHeight(0);
  //std::cerr << "width: " << width << " height: " << height << std::endl;
  auto nofPix = width * height;
  std::vector<float>data(nofPix);
  //glGetTextureImage(gBuffer->depth->getId(),0,GL_DEPTH_COMPONENT,GL_FLOAT,sizeof(float)*data.size(),data.data());
  start(vars,"glGetTextureImage");
  glGetTextureImage(depth->getId(),0,GL_DEPTH_COMPONENT,GL_FLOAT,sizeof(float)*data.size(),data.data());
  stop(vars,"glGetTextureImage");
  float mmin = 10e10;
  float mmax = -10e10;
  for(auto const&p:data){
    mmin = glm::min(mmin,p);
    mmax = glm::max(mmax,p);
  }
  std::cerr << "mmin: " << mmin << " - mmax: " << mmax << std::endl;
}
