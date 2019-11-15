#include<ShadowVolumes.h>

#include<Deferred.h>
#include<Model.h>
#include<Barrier.h>

using namespace std;
using namespace ge::gl;

void shadowVolumesCreateFBO(vars::Vars&vars){
  if(notChanged(vars,"shadowVolumes",__FUNCTION__,{"gBuffer"}))return;

  auto depth = vars.get<GBuffer>("gBuffer")->depth;
  auto fbo = vars.reCreate<Framebuffer>("shadowVolumes.fbo");
  fbo->attachTexture(GL_DEPTH_ATTACHMENT,depth);
  fbo->attachTexture(GL_STENCIL_ATTACHMENT,depth);
  assert(fbo->check());
}

void shadowVolumesCreateMaskFBO(vars::Vars&vars){
  if(notChanged(vars,"shadowVolumes",__FUNCTION__,{"gBuffer","shadowMask"}))return;

  auto depth = vars.get<GBuffer>("gBuffer")->depth;
  auto maskFbo = vars.reCreate<Framebuffer>("shadowVolumes.maskFbo");
  maskFbo->attachTexture(GL_STENCIL_ATTACHMENT,depth);
  maskFbo->attachTexture(GL_COLOR_ATTACHMENT0,vars.get<Texture>("shadowMask"));
  maskFbo->drawBuffers(1,GL_COLOR_ATTACHMENT0);
  assert(maskFbo->check());
}

void shadowVolumesCreateVao(vars::Vars&vars){
  if(notChanged(vars,"shadowVolumes",__FUNCTION__,{}))return;

  vars.reCreate<VertexArray>("shadowVolumes.emptyVao");
}

void shadowVolumesCreateProgram(vars::Vars&vars){
  if(notChanged(vars,"shadowVolumes",__FUNCTION__,{}))return;

#include<ShadowVolumesShaders.h>

  vars.reCreate<Program>("shadowVolumes.program",
      std::make_shared<Shader>(GL_VERTEX_SHADER  ,convertStencilBufferToShadowMaskVPSrc),
      std::make_shared<Shader>(GL_FRAGMENT_SHADER,convertStencilBufferToShadowMaskFPSrc));
}

ShadowVolumes::ShadowVolumes(vars::Vars&vars):ShadowMethod(vars)
{
  shadowVolumesCreateFBO(vars);
  shadowVolumesCreateMaskFBO(vars);
  shadowVolumesCreateVao(vars);
  shadowVolumesCreateProgram(vars);
}

ShadowVolumes::~ShadowVolumes(){
  vars.erase("shadowVolumes");
}

void ShadowVolumes::convertStencilBufferToShadowMask(){
  shadowVolumesCreateMaskFBO(vars);
  shadowVolumesCreateProgram(vars);
  shadowVolumesCreateVao(vars);

  auto maskFbo = vars.get<Framebuffer>("shadowVolumes.maskFbo" );
  auto vao     = vars.get<VertexArray>("shadowVolumes.emptyVao");
  auto program = vars.get<Program    >("shadowVolumes.program" );

  glDisable(GL_DEPTH_TEST);
  maskFbo->bind();
  glClear(GL_COLOR_BUFFER_BIT);
  glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP);
  glStencilFunc(GL_EQUAL,0,0xff);
  glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
  glDepthFunc(GL_ALWAYS);
  glDepthMask(GL_FALSE);
  
  program->use();
  vao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  vao->unbind();
  maskFbo->unbind();
}


void ShadowVolumes::create(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  assert(this!=nullptr);

  ifExistStamp("");

  shadowVolumesCreateFBO(vars);

  auto fbo   = vars.get<Framebuffer>("shadowVolumes.fbo");
  bool zfail = vars.getBool("zfail");

  fbo->bind();
  glClear(GL_STENCIL_BUFFER_BIT);
  glEnable(GL_STENCIL_TEST);
  glStencilFunc(GL_ALWAYS,0,0);

  if(zfail){
    glStencilOpSeparate(GL_FRONT,GL_KEEP,GL_INCR_WRAP,GL_KEEP);
    glStencilOpSeparate(GL_BACK ,GL_KEEP,GL_DECR_WRAP,GL_KEEP);
  }else{
    glStencilOpSeparate(GL_FRONT,GL_KEEP,GL_KEEP,GL_INCR_WRAP);
    glStencilOpSeparate(GL_BACK ,GL_KEEP,GL_KEEP,GL_DECR_WRAP);
  }
  glDepthFunc(GL_LESS);
  glDepthMask(GL_FALSE);
  glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);

  drawSides(lightPosition,viewMatrix,projectionMatrix);
  ifExistStamp("drawSides");

  if(zfail){
    drawCaps(lightPosition,viewMatrix,projectionMatrix);
    ifExistStamp("drawCaps");
  }
  fbo->unbind();

  convertStencilBufferToShadowMask();

  glDepthFunc(GL_LESS);
  glDisable(GL_STENCIL_TEST);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);

  ifExistStamp("convertStencilBufferToShadowMask");
}

shared_ptr<Adjacency const> createAdjacencyBase(vars::Vars&vars){
  vector<float>vertices;
  vars.get<Model>("model")->getVertices(vertices);

  size_t const nofTriangles = vertices.size() / (verticesPerTriangle*componentsPerVertex3D);
  return  make_shared<Adjacency const>(vertices,vars.getSizeT("maxMultiplicity"));
}

