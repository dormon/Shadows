#include <RSSV/ExtractSilhouettes.h>
#include <glm/gtc/type_ptr.hpp>
#include <geGL/StaticCalls.h>
#include <Simplex.h>
#include <FastAdjacency.h>
#include <util.h>
#include <ShadowVolumes.h>
#include<SilhouetteShaders.h>

using namespace ge::gl;
using namespace std;
using namespace glm;
using namespace rssv;

void ExtractSilhouettes::createProgram(){
#include<RSSV/ExtractSilhouettesShader.h>
#include<BallotShader.h>
  program = make_shared<Program>(
      make_shared<Shader>(GL_COMPUTE_SHADER,
        "#version 450\n",
        Shader::define("WAVEFRONT_SIZE"  ,static_cast<int>(vars.getSizeT("wavefrontSize"            ))),
        Shader::define("WGS"             ,static_cast<int>(vars.getSizeT("rssv.extractSilhouetteWGS"))),
        Shader::define("MAX_MULTIPLICITY",static_cast<int>(vars.getSizeT("maxMultiplicity"          ))),
        silhouetteFunctions,
        ballotSrc,
        extractSilhouettesSrc));
}

void ExtractSilhouettes::createDispatchIndirectBuffer(){
  struct DispatchIndirectCommand{
    uint32_t nofWorkGroupsX = 0;
    uint32_t nofWorkGroupsY = 0;
    uint32_t nofWorkGroupsZ = 0;
  };
  DispatchIndirectCommand cmd;
  cmd.nofWorkGroupsY = 1;
  cmd.nofWorkGroupsZ = 1;
  dispatchIndirect = make_shared<Buffer>(sizeof(DispatchIndirectCommand),&cmd);
}

void ExtractSilhouettes::createEdgesBuffer(){
  auto const adj = createAdjacency(vars);
  nofEdges = adj->getNofEdges();

  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;

  size_t const floatAlign = align(vars.getSizeT("rssv.alignment") , sizeof(float)) / sizeof(float) ;
  size_t const bufferSize = align(adj->getNofEdges(),floatAlign)*floatsPerEdge;
  
  auto const src = adj->getVertices().data();
  std::vector<float>dst(bufferSize);
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*0] = src[adj->getEdgeVertexA(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*1] = src[adj->getEdgeVertexA(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*2] = src[adj->getEdgeVertexA(e)+2];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*3] = src[adj->getEdgeVertexB(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*4] = src[adj->getEdgeVertexB(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dst[e+align(adj->getNofEdges(),floatAlign)*5] = src[adj->getEdgeVertexB(e)+2];
  for(size_t o=0;o<maxNofOppositeVertices;++o)
    for(size_t e=0;e<adj->getNofEdges();++e){
      glm::vec4 plane = glm::vec4(0.f);
      if(o<adj->getNofOpposite(e))
        plane = computePlane(toVec3(src+adj->getEdgeVertexA(e)),toVec3(src+adj->getEdgeVertexB(e)),toVec3(src+adj->getOpposite(e,o)));
      for(size_t k=0;k<componentsPerPlane3D;++k)
        dst[e+align(adj->getNofEdges(),floatAlign)*(6+o*componentsPerPlane3D+k)] = plane[(uint32_t)k];
    }

  edges = std::make_shared<Buffer>(dst);


}

struct Silhouette{
  Vertex3Df A;
  Vertex3Df B;
  float multiplicity;
};

void ExtractSilhouettes::createSilhouettesBuffer(){
  silhouettes=std::make_shared<ge::gl::Buffer>(
      sizeof(Silhouette)*nofEdges,
      nullptr,GL_DYNAMIC_COPY);
  silhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
}

ExtractSilhouettes::ExtractSilhouettes(vars::Vars&vars):vars(vars){
  createProgram();
  createEdgesBuffer();
  createSilhouettesBuffer();
  createDispatchIndirectBuffer();
}

void ExtractSilhouettes::extract(glm::vec4 const&light){
  dispatchIndirect->clear(GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  program
    ->set1ui      ("numEdge"               ,(uint32_t)nofEdges        )
    ->set4fv      ("lightPosition"         ,value_ptr(light))
    ->bindBuffer  ("edges"                 ,edges           )
    ->bindBuffer  ("silhouettes"           ,silhouettes     )
    ->bindBuffer  ("dispatchIndirectBuffer",dispatchIndirect)
    ->dispatch(getDispatchSize(nofEdges,vars.getSizeT("rssv.extractSilhouetteWGS")));
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
