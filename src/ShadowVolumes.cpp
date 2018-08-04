#include<ShadowVolumes.h>

#include<Deferred.h>
#include<Model.h>
#include<StencilBufferToShadowMaskProgram.h>

using namespace std;

ShadowVolumes::ShadowVolumes(vars::Vars&vars):ShadowMethod(vars)
{
  assert(this!=nullptr);
  auto depth = vars.get<GBuffer>("gBuffer")->depth;
  fbo = std::make_shared<ge::gl::Framebuffer>();
  fbo->attachTexture(GL_DEPTH_ATTACHMENT,depth);
  fbo->attachTexture(GL_STENCIL_ATTACHMENT,depth);
  assert(fbo->check());

  maskFbo = std::make_shared<ge::gl::Framebuffer>();
  maskFbo->attachTexture(GL_STENCIL_ATTACHMENT,depth);
  maskFbo->attachTexture(GL_COLOR_ATTACHMENT0,vars.get<ge::gl::Texture>("shadowMask"));
  maskFbo->drawBuffers(1,GL_COLOR_ATTACHMENT0);
  assert(maskFbo->check());

  emptyVao = std::make_shared<ge::gl::VertexArray>();

  stencilBufferToShadowMaskProgram = createStencilBufferToShadowMaskProgram();
}

ShadowVolumes::~ShadowVolumes(){}

void ShadowVolumes::convertStencilBufferToShadowMask(){
  assert(this!=nullptr);
  assert(maskFbo!=nullptr);
  assert(emptyVao!=nullptr);
  glDisable(GL_DEPTH_TEST);
  maskFbo->bind();
  glClear(GL_COLOR_BUFFER_BIT);
  glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP);
  glStencilFunc(GL_EQUAL,0,0xff);
  glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
  glDepthFunc(GL_ALWAYS);
  glDepthMask(GL_FALSE);
  stencilBufferToShadowMaskProgram->use();
  emptyVao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  emptyVao->unbind();
  maskFbo->unbind();
}


void ShadowVolumes::create(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  assert(this!=nullptr);
  assert(fbo!=nullptr);

  ifExistStamp("");

  fbo->bind();
  glClear(GL_STENCIL_BUFFER_BIT);
  glEnable(GL_STENCIL_TEST);
  glStencilFunc(GL_ALWAYS,0,0);

  if(vars.getBool("zfail")){
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

  if(vars.getBool("zfail")){
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

shared_ptr<Adjacency const> createAdjacency(vars::Vars&vars){
  vector<float>vertices;
  vars.get<Model>("model")->getVertices(vertices);

  size_t const nofTriangles = vertices.size() / (verticesPerTriangle*componentsPerVertex3D);
  return  make_shared<Adjacency const>(vertices,vars.getSizeT("maxMultiplicity"));
}

