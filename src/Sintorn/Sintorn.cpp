#include<Sintorn/Sintorn.h>
#include<Sintorn/ComputeTileSizes.h>
#include<Sintorn/ShadowFrusta.h>
#include<Sintorn/HierarchicalDepth.h>
#include<Sintorn/MergeStencil.h>
#include<Sintorn/Rasterize.h>
#include<geGL/StaticCalls.h>
#include<FastAdjacency.h>
#include<sstream>
#include<iomanip>
#include<util.h>
#include<Deferred.h>

const size_t DRAWHDB_BINDING_HDBIMAGE = 0;
const size_t DRAWHDB_BINDING_HDT      = 1;

using namespace std;
using namespace ge::gl;

#include<Barrier.h>


Sintorn::Sintorn(vars::Vars&vars):
  ShadowMethod(vars)
{
  assert(this!=nullptr);

  _shadowMask = vars.get<Texture>("shadowMask");

  vars.addBool("sintorn.useUniformTileDivisibility"   ,false);
  vars.addBool("sintorn.useUniformTileSizeInClipSpace",false);

  computeTileSizes(vars);

  //compile shader programs
#include<Sintorn/Shaders.h>


   _blitProgram = make_shared<Program>(
      make_shared<Shader>(GL_COMPUTE_SHADER  ,blitCompSrc));

  _drawHSTProgram = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,drawHSTVertSrc),
      make_shared<Shader>(GL_FRAGMENT_SHADER,drawHSTFragSrc));

  _drawFinalStencilMask = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,drawHSTVertSrc),
      make_shared<Shader>(GL_FRAGMENT_SHADER,drawFinalStencilMaskFragSrc));


  _emptyVao=make_shared<VertexArray>();

  allocateHierarchicalStencil(vars);

}

Sintorn::~Sintorn(){
}

void Sintorn::create(
    glm::vec4 const&lightPosition,
    glm::mat4 const&view      ,
    glm::mat4 const&projection){
  computeTileSizes(vars);

  assert(this!=nullptr);
  ifExistStamp("");
  //computeHierarchicalDepth(vars,lightPosition);
  ifExistStamp("computeHDT");
  //computeShadowFrusta(vars,lightPosition,projection*view);
  ifExistStamp("computeShadowFrusta");
  //rasterize(vars);
  ifExistStamp("rasterize");
  //mergeStencil(vars);
  ifExistStamp("merge");
  //blit();
  ifExistStamp("blit");
}

void Sintorn::drawHST(size_t l){
  assert(this!=nullptr);
  _drawHSTProgram->use();
  auto&HST = vars.getVector<std::shared_ptr<Texture>>("sintorn.HST");
  HST[l]->bindImage(0);
  _emptyVao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  _emptyVao->unbind();
}

void Sintorn::drawFinalStencilMask(){
  assert(this!=nullptr);
  assert(_drawFinalStencilMask!=nullptr);
  assert(_drawFinalStencilMask!=nullptr);
  assert(_emptyVao!=nullptr);
  _drawFinalStencilMask->use();
  auto finalStencilMask = vars.get<Texture>("sintorn.finalStencilMask");
  finalStencilMask->bindImage(0);
  _emptyVao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  _emptyVao->unbind();
}

void Sintorn::blit(){
  assert(this!=nullptr);
  assert(_blitProgram!=nullptr);
  assert(_shadowMask!=nullptr);
  _blitProgram->use();
  auto finalStencilMask = vars.get<Texture>("sintorn.finalStencilMask");
  finalStencilMask->bindImage(0);
  vars.get<Texture>("shadowMask")->bindImage(1);
  _blitProgram->set2uiv("windowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));
  glDispatchCompute(
      (GLuint)getDispatchSize(vars.get<glm::uvec2>("windowSize")->x,8),
      (GLuint)getDispatchSize(vars.get<glm::uvec2>("windowSize")->y,8),1);
}
