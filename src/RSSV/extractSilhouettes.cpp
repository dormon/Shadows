#include <glm/glm.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <Model.h>
#include <FastAdjacency.h>
#include <divRoundUp.h>
#include <align.h>
#include <perfCounters.h>
#include <ShadowMethod.h>
#include <SilhouetteShaders.h>
#include <BallotShader.h>
#include <util.h>
#include<createAdjacency.h>


#include <RSSV/extractSilhouettes.h>
#include <RSSV/extractSilhouettesShader.h>

using namespace ge::gl;
using namespace std;

namespace rssv{

void createEdgePlanes(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"adjacency"
      ,"rssv.param.alignment"
      );
  auto const adj = vars.get<Adjacency>("adjacency");
  size_t const alignSize = vars.getSizeT ("rssv.param.alignment");

  size_t const verticesPerEdge = 2;
  size_t const componentsPerVertex3D = 3;
  size_t const componentsPerPlane3D = 4;

  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;

  size_t floatAlign = align(alignSize , sizeof(float));
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

  vars.reCreate<Buffer>("rssv.method.edgePlanes",dst);
}

void createEdges(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"adjacency"
      );

  auto adj = vars.get<Adjacency>("adjacency");
  std::cerr << "nofEdges: " << adj->getNofEdges() << std::endl;
  auto&vert = adj->getVertices();

  auto nofE = adj->getNofEdges();
  auto anofE = divRoundUp(nofE,1024)*1024;

  std::vector<float>edges;
  edges.resize(anofE*6,0);
  for(size_t e=0;e<adj->getNofEdges();++e)edges[e+0*anofE] = vert[adj->getEdgeVertexA(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)edges[e+1*anofE] = vert[adj->getEdgeVertexA(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)edges[e+2*anofE] = vert[adj->getEdgeVertexA(e)+2];
  for(size_t e=0;e<adj->getNofEdges();++e)edges[e+3*anofE] = vert[adj->getEdgeVertexB(e)+0];
  for(size_t e=0;e<adj->getNofEdges();++e)edges[e+4*anofE] = vert[adj->getEdgeVertexB(e)+1];
  for(size_t e=0;e<adj->getNofEdges();++e)edges[e+5*anofE] = vert[adj->getEdgeVertexB(e)+2];
  vars.reCreate<uint32_t>("rssv.method.alignedNofEdges",anofE);
  
  vars.reCreate<Buffer>("rssv.method.edgeBuffer",edges);
}

void allocateSilhouettesData(vars::Vars&vars){
  createEdgePlanes(vars);
  createEdges(vars);
}

void createSilhouetteProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"         ,
      "rssv.param.alignment"              ,
      "rssv.param.extractSilhouettesWGS"  ,
      "maxMultiplicity"                   ,
      "wavefrontSize"                     ,
      "adjacency"                         );

  auto adj = vars.get<Adjacency>("adjacency");
  vars.reCreate<Program>("rssv.method.extractSilhouettesProgram",
      make_shared<Shader>(GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("WARP"                     ,uint32_t( vars.getSizeT ("wavefrontSize"                     ))),
        Shader::define("ALIGN_SIZE"               ,uint32_t( vars.getSizeT ("rssv.param.alignment"              ))),
        Shader::define("WORKGROUP_SIZE_X"         ,int32_t ( vars.getUint32("rssv.param.extractSilhouettesWGS"  ))),
        Shader::define("MAX_MULTIPLICITY"         ,int32_t ( vars.getUint32("maxMultiplicity"                   ))),
        Shader::define("NOF_EDGES"                ,uint32_t( adj->getNofEdges()                                  )),
        ballotSrc,
        silhouetteFunctions,
        extractSilhouettesShader));
}

void allocateSilhouetteCounter(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method");
  vars.reCreate<Buffer>("rssv.method.silhouetteCounter",sizeof(uint32_t)*4);
}

void allocateMultBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method"
      ,"adjacency");

  auto const adj = vars.get<Adjacency>("adjacency");

  vars.reCreate<Buffer>("rssv.method.multBuffer",sizeof(uint32_t)*adj->getNofEdges());
}

void extractSilhouettes(vars::Vars&vars){
  FUNCTION_CALLER();
  createAdjacency(vars);
  allocateSilhouettesData(vars);
  createSilhouetteProgram(vars);
  allocateSilhouetteCounter(vars);
  allocateMultBuffer(vars);

  auto silhouetteCounter =  vars.get<Buffer>   ("rssv.method.silhouetteCounter"        );
  auto adj               =  vars.get<Adjacency>("adjacency"                            );
  auto program           =  vars.get<Program>  ("rssv.method.extractSilhouettesProgram");
  auto edgePlanes        =  vars.get<Buffer>   ("rssv.method.edgePlanes"               );
  auto WGS               =  vars.getUint32     ("rssv.param.extractSilhouettesWGS"     );
  auto multBuffer        =  vars.get<Buffer>   ("rssv.method.multBuffer"               );
  auto lightPosition     = *vars.get<glm::vec4>("rssv.method.lightPosition"            );

  silhouetteCounter->clear(GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  program
    ->set4fv    ("lightPosition"     ,glm::value_ptr(lightPosition))
    ->bindBuffer("EdgePlanes"        ,edgePlanes                   )
    ->bindBuffer("DrawIndirectBuffer",silhouetteCounter            )
    ->bindBuffer("MultBuffer"        ,multBuffer                   )
    ->dispatch((GLuint)getDispatchSize(adj->getNofEdges(),WGS));

  glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

}
