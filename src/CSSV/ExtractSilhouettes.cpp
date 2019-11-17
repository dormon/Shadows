#include <CSSV/ExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <util.h>
#include<CSSV/ExtractSilhouetteShader.h>
#include<SilhouetteShaders.h>
#include<FunctionPrologue.h>
#include<CSSV/createExtractProgram.h>

using namespace std;
using namespace ge::gl;
using namespace cssv;

shared_ptr<Buffer>createDIBO(){
  struct DrawArraysIndirectCommand{
    uint32_t nofVertices  = 0;
    uint32_t nofInstances = 0;
    uint32_t firstVertex  = 0;
    uint32_t baseInstance = 0;
  };
  DrawArraysIndirectCommand cmd;
  cmd.nofInstances = 1;
  return make_shared<Buffer>(sizeof(DrawArraysIndirectCommand),&cmd);
}

ExtractSilhouettes::ExtractSilhouettes(vars::Vars&vars):vars(vars){
  dibo = createDIBO();
}

void ExtractSilhouettes::compute(glm::vec4 const&lightPosition){
  cssv::createExtractProgram(vars);
  assert(this                      !=nullptr);
  dibo->clear(GL_R32UI,0,sizeof(uint32_t),GL_RED_INTEGER,GL_UNSIGNED_INT);

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  auto program = vars.get<Program>("cssv.method.extractProgram");

  program
    ->set4fv    ("lightPosition"     ,glm::value_ptr(lightPosition))
    ->bindBuffer("Edges"             ,edges                        )
    ->bindBuffer("Silhouettes"       ,sillhouettes                 )
    ->bindBuffer("DrawIndirectBuffer",dibo                         )
    ->dispatch((GLuint)getDispatchSize(nofEdges,vars.getUint32("cssv.param.computeSidesWGS")));

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  glFinish();

}

