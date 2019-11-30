#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>

#include <Sintorn2/buildHierarchy.h>
#include <Sintorn2/allocateHierarchy.h>
#include <Sintorn2/createBuildHierarchyProgram.h>

using namespace ge::gl;

void sintorn2::buildHierarchy(vars::Vars&vars){
  sintorn2::allocateHierarchy(vars);
  sintorn2::createBuildHierarchyProgram(vars);

  auto depth       = vars.get<GBuffer>("gBuffer")->depth;
  auto prg         = vars.get<Program>("sintorn2.method.buildHierarchyProgram");
  auto nodePool    = vars.get<Buffer >("sintorn2.method.nodePool");
  auto aabbPool    = vars.get<Buffer >("sintorn2.method.aabbPool");
  auto aabbCounter = vars.get<Buffer >("sintorn2.method.aabbCounter");

  aabbPool   ->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  aabbCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);

  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  aabbCounter->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  
  depth->bind(1);
  
  prg->use();
  auto const clusterX = vars.getUint32("sintorn2.clusterX");
  auto const clusterY = vars.getUint32("sintorn2.clusterY");
  glDispatchCompute(clusterX,clusterY,1);

  //std::vector<uint32_t>ddd;
  //hierarchy->getData(ddd);
  //std::cerr << ddd[0] << std::endl;
  //std::cerr << ddd[1] << std::endl;

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
