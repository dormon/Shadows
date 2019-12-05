#include <Vars/Vars.h>
#include <CSSV/sides/extractSilhouettes.h>
#include <CSSV/sides/createExtractProgram.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <FastAdjacency.h>
#include <glm/gtc/type_ptr.hpp>
#include <util.h>

using namespace ge::gl;

void cssv::sides::extractSilhouettes(vars::Vars&vars,glm::vec4 const&lightPosition){
  createExtractProgram(vars);

  auto dibo        = vars.get<Buffer>   ("cssv.method.dibo"          );
  auto adj         = vars.get<Adjacency>("adjacency"                 );
  auto program     = vars.get<Program>  ("cssv.method.extractProgram");
  auto edges       = vars.get<Buffer>   ("cssv.method.edges"         );
  auto silhouettes = vars.get<Buffer>   ("cssv.method.silhouettes"   );
  auto WGS         = vars.getUint32     ("cssv.param.computeSidesWGS");

  dibo->clear(GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


  if(!vars.getBool("cssv.param.dontExtractMultiplicity")){
    auto multBuffer  = vars.get<Buffer>   ("cssv.method.multBuffer");
    multBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,3);
  }

  program
    ->set4fv    ("lightPosition"     ,glm::value_ptr(lightPosition))
    ->bindBuffer("Edges"             ,edges                        )
    ->bindBuffer("Silhouettes"       ,silhouettes                  )
    ->bindBuffer("DrawIndirectBuffer",dibo                         )
    ->dispatch((GLuint)getDispatchSize(adj->getNofEdges(),WGS));

  /*
  std::vector<uint32_t>diboData;
  dibo->getData(diboData);
  for(auto const&x:diboData)
    std::cerr << x << " ";
  std::cerr << std::endl;
  */

  /*
  std::vector<float>edgesData;
  edges->getData(edgesData);
  for(auto const&x:edgesData)
    std::cerr << x << " ";
  std::cerr << std::endl;
  */


  glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}
