#include<CubeShadowMapping/CubeShadowMapping.h>
#include<Deferred.h>
#include<Model.h>

using namespace ge::gl;
using namespace std;

#include<Vars/Resource.h>

#include<FunctionPrologue.h>

void createShadowMapTexture(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm","args.csm.resolution");

  auto const resolution = vars.getUint32("args.csm.resolution");
  auto shadowMap = vars.reCreate<Texture>("csm.shadowMap",GL_TEXTURE_CUBE_MAP,GL_DEPTH_COMPONENT24,1,resolution,resolution);
  shadowMap->texParameteri(GL_TEXTURE_MIN_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_MAG_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_S      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_T      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL              );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_R_TO_TEXTURE);
}

void createShadowMapFBO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm","csm.shadowMap");

  auto fbo = vars.reCreate<Framebuffer>("csm.shadowMapFBO");
  fbo->attachTexture(GL_DEPTH_ATTACHMENT,vars.get<Texture>("csm.shadowMap"));
}


void createShadowMapVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm","renderModel");

  auto vao = vars.reCreate<VertexArray>("csm.shadowMapVAO");
  vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices,0,3,GL_FLOAT);
}

void createShadowMapProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm");

#include<CubeShadowMapping/CreateShadowMapShaders.h>
  vars.reCreate<Program>("csm.shadowMapProgram",
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMapVertexShaderSource  ),
      make_shared<Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        createShadowMapGeometryShaderSource));
}

void createMaskFBO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm","shadowMask");

  auto fbo = vars.reCreate<Framebuffer>("csm.maskFBO");
  fbo->attachTexture(GL_COLOR_ATTACHMENT0,vars.get<Texture>("shadowMask"));
  fbo->drawBuffers(1,GL_COLOR_ATTACHMENT0);
}

void createShadowMaskProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm");

#include<CubeShadowMapping/ShadowMapToShadowMaskShaders.h>

  vars.reCreate<Program>("csm.shadowMaskProgram",
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMaskVertexShaderSource),
      make_shared<Shader>(GL_FRAGMENT_SHADER,
        "#version 450\n",
        createShadowMaskFragmentShaderSource));
}

void createShadowMaskVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("csm");

  vars.reCreate<VertexArray>("csm.maskVAO");
}

CubeShadowMapping::CubeShadowMapping(
        vars::Vars&vars       ):
  ShadowMethod(vars)
{
}

CubeShadowMapping::~CubeShadowMapping(){
  vars.erase("csm");
}

void CubeShadowMapping::fillShadowMap(glm::vec4 const&lightPosition){
  createShadowMapTexture(vars);
  createShadowMapFBO(vars);
  createShadowMapVAO(vars);
  createShadowMapProgram(vars);

  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(2.5,10);
  auto const resolution = vars.getUint32("args.csm.resolution");
  auto const near       = vars.getFloat ("args.csm.near"      );
  auto const far        = vars.getFloat ("args.csm.far"       );
  auto const faces      = vars.getUint32("args.csm.faces"     );
  glViewport(0,0,resolution,resolution);
  glEnable(GL_DEPTH_TEST);
  vars.get<Framebuffer>("csm.shadowMapFBO")->bind();
  glClear(GL_DEPTH_BUFFER_BIT);
  vars.get<VertexArray>("csm.shadowMapVAO")->bind();
  vars.get<Program>("csm.shadowMapProgram")
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  glDrawArraysInstanced(GL_TRIANGLES,0,vars.get<RenderModel>("renderModel")->nofVertices,faces);
  vars.get<VertexArray>("csm.shadowMapVAO")->unbind();
  vars.get<Framebuffer>("csm.shadowMapFBO")->unbind();
}

void CubeShadowMapping::fillShadowMask(glm::vec4 const&lightPosition){
  createShadowMaskVAO(vars);
  createMaskFBO(vars);
  createShadowMaskProgram(vars);

  auto const near = vars.getFloat("args.csm.near");
  auto const far = vars.getFloat("args.csm.far");
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  glViewport(0,0,windowSize.x,windowSize.y);
  vars.get<Framebuffer>("csm.maskFBO")->bind();
  vars.get<VertexArray>("csm.maskVAO")->bind();
  vars.get<Program>("csm.shadowMaskProgram")
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  vars.get<GBuffer>("gBuffer")->position->bind(0);
  vars.get<Texture>("csm.shadowMap")->bind(1);
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  vars.get<VertexArray>("csm.maskVAO")->unbind();
  vars.get<Framebuffer>("csm.maskFBO")->unbind();
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

