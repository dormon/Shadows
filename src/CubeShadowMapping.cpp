#include<CubeShadowMapping.h>
#include<Deferred.h>
#include<Model.h>

using namespace ge::gl;
using namespace std;

shared_ptr<Texture>createShadowMapTexture(vars::Vars&vars){
  auto shadowMap = make_shared<Texture>(GL_TEXTURE_CUBE_MAP,GL_DEPTH_COMPONENT24,1,vars.getUint32("csm.resolution"),vars.getUint32("csm.resolution"));
  shadowMap->texParameteri(GL_TEXTURE_MIN_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_MAG_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_S      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_T      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL              );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_R_TO_TEXTURE);
  return shadowMap;
}

shared_ptr<Framebuffer>createMaskFBO(vars::Vars&vars){
  auto fbo = make_shared<Framebuffer>();
  fbo->attachTexture(GL_COLOR_ATTACHMENT0,vars.get<Texture>("shadowMask"));
  fbo->drawBuffers(1,GL_COLOR_ATTACHMENT0);
  return fbo;
}

shared_ptr<Program>createShadowMapProgram(){
#include<CubeShadowMappingShaders.h>

  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMapVertexShaderSource  ),
      make_shared<Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        createShadowMapGeometryShaderSource));
  return program;
}

shared_ptr<Program>createShadowMaskProgram(){
#include<CubeShadowMappingShaders.h>

  auto program = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMaskVertexShaderSource),
      make_shared<Shader>(GL_FRAGMENT_SHADER,
        "#version 450\n",
        createShadowMaskFragmentShaderSource));
  return program;
}

shared_ptr<Framebuffer>createShadowMapFBO(shared_ptr<Texture>const&shadowMap){
  auto fbo = make_shared<Framebuffer>();
  fbo->attachTexture(GL_DEPTH_ATTACHMENT,shadowMap);
  return fbo;
}

shared_ptr<VertexArray>createShadowMapVAO(vars::Vars&vars){
  auto vao = make_shared<VertexArray>();
  vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices,0,3,GL_FLOAT);
  return vao;
}

shared_ptr<VertexArray>createShadowMaskVAO(){
  return make_shared<VertexArray>();
}

CubeShadowMapping::CubeShadowMapping(
        vars::Vars&vars       ):
  ShadowMethod(vars)
{
  shadowMap        = createShadowMapTexture(vars);
  shadowMapFBO     = createShadowMapFBO(shadowMap);
  shadowMapVAO     = createShadowMapVAO(vars);
  maskVao          = createShadowMaskVAO();
  maskFbo          = createMaskFBO(vars);
  createShadowMap  = createShadowMapProgram();
  createShadowMask = createShadowMaskProgram();
}

CubeShadowMapping::~CubeShadowMapping(){}

void CubeShadowMapping::fillShadowMap(glm::vec4 const&lightPosition){
  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(2.5,10);
  auto const resolution = vars.getUint32("csm.resolution");
  auto const near = vars.getFloat("csm.near");
  auto const far = vars.getFloat("csm.far");
  auto const faces = vars.getUint32("csm.faces");
  glViewport(0,0,resolution,resolution);
  glEnable(GL_DEPTH_TEST);
  shadowMapFBO->bind();
  glClear(GL_DEPTH_BUFFER_BIT);
  shadowMapVAO->bind();
  createShadowMap
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  glDrawArraysInstanced(GL_TRIANGLES,0,vars.get<RenderModel>("renderModel")->nofVertices,faces);
  shadowMapVAO->unbind();
  shadowMapFBO->unbind();
}

void CubeShadowMapping::fillShadowMask(glm::vec4 const&lightPosition){
  auto const near = vars.getFloat("csm.near");
  auto const far = vars.getFloat("csm.far");
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  glViewport(0,0,windowSize.x,windowSize.y);
  maskFbo->bind();
  maskVao->bind();
  createShadowMask
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  vars.get<GBuffer>("gBuffer")->position->bind(0);
  shadowMap->bind(1);
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  maskVao->unbind();
  maskFbo->unbind();
}

void CubeShadowMapping::create(
    glm::vec4 const&lightPosition,
    glm::mat4 const&             ,
    glm::mat4 const&             ){
  ifExistStamp("");
  fillShadowMap(lightPosition);
  ifExistStamp("createShadowMap");
  fillShadowMask(lightPosition);
  ifExistStamp("createShadowMask");
}

