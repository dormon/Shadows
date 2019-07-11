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
