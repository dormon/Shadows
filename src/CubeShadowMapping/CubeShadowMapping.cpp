#include<CubeShadowMapping/CubeShadowMapping.h>
#include<Deferred.h>
#include<Model.h>

using namespace ge::gl;
using namespace std;

#include<Vars/Resource.h>

#include<FunctionPrologue.h>

void createShadowMapTexture(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm.method","csm.param.resolution");

  auto const resolution = vars.getUint32("csm.param.resolution");
  auto shadowMap = vars.reCreate<Texture>("csm.method.shadowMap",GL_TEXTURE_CUBE_MAP,GL_DEPTH_COMPONENT24,1,resolution,resolution);
  shadowMap->texParameteri(GL_TEXTURE_MIN_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_MAG_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_S      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_T      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL              );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_R_TO_TEXTURE);
}

void createShadowMapFBO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm.method","csm.method.shadowMap");

  auto fbo = vars.reCreate<Framebuffer>("csm.method.shadowMapFBO");
  fbo->attachTexture(GL_DEPTH_ATTACHMENT,vars.get<Texture>("csm.method.shadowMap"));
}


void createShadowMapVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm.method","renderModel");

  auto vao = vars.reCreate<VertexArray>("csm.method.shadowMapVAO");
  vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices,0,3,GL_FLOAT);
}

void createShadowMapProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm.method");

#include<CubeShadowMapping/CreateShadowMapShaders.h>
  vars.reCreate<Program>("csm.method.shadowMapProgram",
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMapVertexShaderSource  ),
      make_shared<Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        createShadowMapGeometryShaderSource),
      make_shared<Shader>(GL_FRAGMENT_SHADER,
        "#version 450\n",
        createShadowMapFragmentShaderSource)
      
      );
}

void createMaskFBO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm.method","shadowMask");

  auto fbo = vars.reCreate<Framebuffer>("csm.method.maskFBO");
  fbo->attachTexture(GL_COLOR_ATTACHMENT0,vars.get<Texture>("shadowMask"));
  fbo->drawBuffers(1,GL_COLOR_ATTACHMENT0);
}

void createShadowMaskProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm.method");

#include<CubeShadowMapping/ShadowMapToShadowMaskShaders.h>

  vars.reCreate<Program>("csm.method.shadowMaskProgram",
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMaskVertexShaderSource),
      make_shared<Shader>(GL_FRAGMENT_SHADER,
        "#version 450\n",
        createShadowMaskFragmentShaderSource));
}

void createShadowMaskVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm.method");

  vars.reCreate<VertexArray>("csm.method.maskVAO");
}

CubeShadowMapping::CubeShadowMapping(
        vars::Vars&vars       ):
  ShadowMethod(vars)
{
}

CubeShadowMapping::~CubeShadowMapping(){
  vars.erase("csm.method");
}

void CubeShadowMapping::fillShadowMap(glm::vec4 const&lightPosition){
  createShadowMapTexture(vars);
  createShadowMapFBO(vars);
  createShadowMapVAO(vars);
  createShadowMapProgram(vars);

  glEnable(GL_POLYGON_OFFSET_FILL);
  auto const factor      = vars.getFloat ("csm.param.factor"     );
  auto const units       = vars.getFloat ("csm.param.units"      );
  glPolygonOffset(factor,units);
  auto const resolution = vars.getUint32("csm.param.resolution");
  auto const near       = vars.getFloat ("csm.param.near"      );
  auto const far        = vars.getFloat ("csm.param.far"       );
  auto const faces      = vars.getUint32("csm.param.faces"     );
  auto const fbo        = vars.get<Framebuffer>("csm.method.shadowMapFBO");
  auto const vao        = vars.get<VertexArray>("csm.method.shadowMapVAO");
  auto const program    = vars.get<Program>("csm.method.shadowMapProgram");


  glViewport(0,0,resolution,resolution);
  glEnable(GL_DEPTH_TEST);
  fbo->bind();
  glClear(GL_DEPTH_BUFFER_BIT);
  vao->bind();
  program
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  glDrawArraysInstanced(GL_TRIANGLES,0,vars.get<RenderModel>("renderModel")->nofVertices,faces);
  vao->unbind();
  fbo->unbind();
}

void CubeShadowMapping::fillShadowMask(glm::vec4 const&lightPosition){
  createShadowMaskVAO(vars);
  createMaskFBO(vars);
  createShadowMaskProgram(vars);

  auto const near = vars.getFloat("csm.param.near");
  auto const far =  vars.getFloat("csm.param.far");
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  glViewport(0,0,windowSize.x,windowSize.y);
  vars.get<Framebuffer>("csm.method.maskFBO")->bind();
  vars.get<VertexArray>("csm.method.maskVAO")->bind();
  vars.get<Program    >("csm.method.shadowMaskProgram")
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  vars.get<GBuffer>("gBuffer")->position->bind(0);
  vars.get<Texture>("csm.method.shadowMap")->bind(1);
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  vars.get<VertexArray>("csm.method.maskVAO")->unbind();
  vars.get<Framebuffer>("csm.method.maskFBO")->unbind();
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

