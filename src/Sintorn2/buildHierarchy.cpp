#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>

#include <Sintorn2/buildHierarchy.h>
#include <Sintorn2/allocateHierarchy.h>
#include <Sintorn2/createBuildHierarchyProgram.h>

using namespace ge::gl;

#include <iomanip>

void sintorn2::buildHierarchy(vars::Vars&vars){
  FUNCTION_CALLER();

  sintorn2::allocateHierarchy(vars);
  sintorn2::createBuildHierarchyProgram(vars);
  //exit(0);
  auto depth       = vars.get<GBuffer>("gBuffer")->depth;
  auto prg         = vars.get<Program>("sintorn2.method.buildHierarchyProgram");
  auto nodePool    = vars.get<Buffer >("sintorn2.method.nodePool");
  auto aabbPool    = vars.get<Buffer >("sintorn2.method.aabbPool");
  auto aabbCounter = vars.get<Buffer >("sintorn2.method.aabbCounter");

  aabbPool   ->clear(GL_R32F ,GL_RED        ,GL_FLOAT       );
  aabbCounter->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);
  nodePool   ->clear(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT);

  nodePool   ->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  aabbPool   ->bindBase(GL_SHADER_STORAGE_BUFFER,1);
  aabbCounter->bindBase(GL_SHADER_STORAGE_BUFFER,2);
  
  depth->bind(1);
  
  prg->use();
  auto const clustersX = vars.getUint32("sintorn2.method.clustersX");
  auto const clustersY = vars.getUint32("sintorn2.method.clustersY");
  glDispatchCompute(clustersX,clustersY,1);

  ////std::vector<float>nodePoolData;
  //std::vector<uint32_t>nodePoolData;
  //nodePool->getData(nodePoolData);

  //glFinish();
  ////for(size_t i=0;i<1000;++i)
  ////  std::cerr << nodePoolData[1+32+32*32+32*32*32+i] << std::endl;

  //for(auto const&n:nodePoolData)
  //  std::cerr << n << std::endl;
  //exit(0);

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
