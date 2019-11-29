#include <Sintorn2/Sintorn2.h>
#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>
#include <startStop.h>
#include <sstream>
#include <algorithm>
#include <BallotShader.h>

#include <Sintorn2/allocateHierarchy.h>
#include <Sintorn2/createBuildHierarchyProgram.h>

Sintorn2::Sintorn2(vars::Vars& vars) : ShadowMethod(vars) {}

Sintorn2::~Sintorn2() {vars.erase("cssv.method");}



void Sintorn2::create(glm::vec4 const& lightPosition,
                      glm::mat4 const& viewMatrix,
                      glm::mat4 const& projectionMatrix)
{
  sintorn2::allocateHierarchy(vars);
  sintorn2::createBuildHierarchyProgram(vars);

  auto prg = vars.get<ge::gl::Program>("sintorn2.hierarchyProgram0");

  auto depth = vars.get<GBuffer>("gBuffer")->depth;
  auto hierarchy = vars.get<ge::gl::Buffer>("sintorn2.hierarchy");
  hierarchy->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  prg->bindBuffer("Hierarchy",hierarchy);
  depth->bind(1);
  

  prg->use();
  auto const clusterX = vars.getUint32("sintorn2.clusterX");
  auto const clusterY = vars.getUint32("sintorn2.clusterY");
  glDispatchCompute(clusterX,clusterY,1);
  std::vector<uint32_t>ddd;
  hierarchy->getData(ddd);
  std::cerr << ddd[0] << std::endl;
  std::cerr << ddd[1] << std::endl;

  /*
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

  // */
}
