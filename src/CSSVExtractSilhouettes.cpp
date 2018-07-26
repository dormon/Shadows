#include <CSSVExtractSilhouettes.h>
#include <FastAdjacency.h>
#include <geGL/StaticCalls.h>
#include <glm/gtc/type_ptr.hpp>
#include <util.h>

using namespace std;
using namespace ge::gl;

CSSVExtractSilhouettes::CSSVExtractSilhouettes(vars::Vars&vars,shared_ptr<Adjacency const>const&adj):vars(vars){
#include<CSSVExtractSilhouetteShader.h>
#include<SilhouetteShaders.h>
  program = make_shared<Program>(
      make_shared<Shader>(GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("WORKGROUP_SIZE_X",int32_t(vars.getBool("cssv.computeSidesWGS"))),
        Shader::define("MAX_MULTIPLICITY",int32_t(adj->getMaxMultiplicity()           )),
        Shader::define("LOCAL_ATOMIC"    ,int32_t(vars.getBool("cssv.localAtomic"    ))),
        Shader::define("CULL_SIDES"      ,int32_t(vars.getBool("cssv.cullSides"      ))),
        Shader::define("USE_PLANES"      ,int32_t(vars.getBool("cssv.usePlanes"      ))),
        Shader::define("USE_INTERLEAVING",int32_t(vars.getBool("cssv.useInterleaving"))),
        silhouetteFunctions,
        computeSrc));
}

void CSSVExtractSilhouettes::compute(glm::vec4 const&lightPosition){
  assert(this                      !=nullptr);
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

