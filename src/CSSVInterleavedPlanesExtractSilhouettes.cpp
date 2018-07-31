#include <CSSVInterleavedPlanesExtractSilhouettes.h>
#include <geGL/StaticCalls.h>
#include <util.h>
#include <FastAdjacency.h>
#include <ShadowMethod.h>

using namespace ge::gl;
using namespace std;

size_t align(size_t what,size_t alignment){
  return (what / alignment) * alignment + (size_t)((what % alignment)!=0)*alignment;
}

CSSVInterleavedPlanesExtractSilhouettes::CSSVInterleavedPlanesExtractSilhouettes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj):CSSVExtractSilhouettes(vars,adj){
#include<CSSVInterleavedPlanesShader.h>
#include<SilhouetteShaders.h>
  program = make_shared<Program>(
      make_shared<Shader>(GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("WORKGROUP_SIZE_X",int32_t(vars.getUint32("cssv.computeSidesWGS"))),
        Shader::define("MAX_MULTIPLICITY",int32_t(adj->getMaxMultiplicity()             )),
        Shader::define("LOCAL_ATOMIC"    ,int32_t(vars.getBool("cssv.localAtomic"      ))),
        Shader::define("CULL_SIDES"      ,int32_t(vars.getBool("cssv.cullSides"        ))),
        Shader::define("USE_PLANES"      ,int32_t(vars.getBool("cssv.usePlanes"        ))),
        Shader::define("USE_INTERLEAVING",int32_t(vars.getBool("cssv.useInterleaving"  ))),
        silhouetteFunctions,
        computeSrc));



  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;

  size_t const alignSize = 128;
  size_t const bufferSize = adj->getNofEdges()*floatsPerEdge*sizeof(float);
  size_t const alignBufferSize = align(bufferSize,alignSize);
  
  size_t const floatAlign = alignSize / sizeof(float);

  edges = std::make_shared<Buffer>(alignBufferSize);

  auto const srcPtr = static_cast<float const*>(adj->getVertices() );

  std::vector<float>dstPtr;
  dstPtr.resize(alignBufferSize/sizeof(float));
  //auto const dstPtr = static_cast<float      *>(edges->map());
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+align(adj->getNofEdges()*0,floatAlign)] = srcPtr[adj->getEdgeVertexA(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+align(adj->getNofEdges()*1,floatAlign)] = srcPtr[adj->getEdgeVertexA(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+align(adj->getNofEdges()*2,floatAlign)] = srcPtr[adj->getEdgeVertexA(e)+2];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+align(adj->getNofEdges()*3,floatAlign)] = srcPtr[adj->getEdgeVertexB(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+align(adj->getNofEdges()*4,floatAlign)] = srcPtr[adj->getEdgeVertexB(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)dstPtr[e+align(adj->getNofEdges()*5,floatAlign)] = srcPtr[adj->getEdgeVertexB(e)+2];
  for(size_t o=0;o<maxNofOppositeVertices;++o)
    for(size_t e=0;e<adj->getNofEdges();++e){
      glm::vec4 plane = glm::vec4(0.f);
      if(o<adj->getNofOpposite(e))
        plane = computePlane(toVec3(srcPtr+adj->getEdgeVertexA(e)),toVec3(srcPtr+adj->getEdgeVertexB(e)),toVec3(srcPtr+adj->getOpposite(e,o)));
      for(size_t k=0;k<componentsPerPlane3D;++k)
        dstPtr[e+align(adj->getNofEdges()*(6+o*componentsPerPlane3D+k),floatAlign)] = plane[(uint32_t)k];
    }
  //edges->unmap();
  edges->setData(dstPtr.data());


  nofEdges = adj->getNofEdges();

  sillhouettes=std::make_shared<ge::gl::Buffer>(
      sizeof(float)*componentsPerVertex4D*verticesPerQuad*nofEdges*adj->getMaxMultiplicity(),
      nullptr,GL_DYNAMIC_COPY);
  sillhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);
}

void CSSVInterleavedPlanesExtractSilhouettes::compute(glm::vec4 const&lightPosition){
  auto const bufferSize = edges->getSize();
  auto const gigabyte = 1024*1024*1024;
  auto const bufferSizeInGigabytes = static_cast<double>(bufferSize) / static_cast<double>(gigabyte);
  dibo->clear(GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  program
    ->set1ui    ("numEdge"           ,uint32_t(nofEdges)    )
    ->set4fv    ("lightPosition"     ,glm::value_ptr(lightPosition))
    ->bindBuffer("edges"             ,edges                 )
    ->bindBuffer("silhouettes"       ,sillhouettes          )
    ->bindBuffer("drawIndirectBuffer",dibo                  )
    ->dispatch((GLuint)getDispatchSize(nofEdges,vars.getUint32("cssv.computeSidesWGS")));
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();

}
