#include<CubeShadowMapping.h>
#include<Deferred.h>
#include<Model.h>

using namespace ge::gl;
using namespace std;

#include<Vars/Resource.h>
class Barrier{
  public:
    Barrier(vars::Vars&vars,std::vector<std::string>const&inputs = {},std::vector<std::string>const&outputs = {}):vars(vars){
      for(auto const&i:inputs){
        if(!vars.has(i))
          throw std::runtime_error(std::string("cannot create Barrier, missing input variable: ")+i);
        resources.push_back(std::tuple<std::shared_ptr<vars::Resource>,size_t>(vars.getResource(i),vars.getTicks(i)));
      }
    }
    bool notChange(){
      bool changed = firstCall;
      firstCall = false;
      for(auto const&r:resources)
        if(std::get<0>(r)->getTicks() > std::get<1>(r)){
          changed |= true;
          break;
        }
      if(changed)
        for(auto &r:resources)
          std::get<1>(r) = std::get<0>(r)->getTicks();
      return !changed;
    }
  protected:
    vars::Vars&vars;
    std::vector<std::tuple<std::shared_ptr<vars::Resource>,size_t>>resources;
    bool firstCall = true;

};

void createShadowMapTexture(vars::Vars&vars){
  static Barrier barrier(vars,{"csm.resolution"});
  if(barrier.notChange())return;
  std::cerr << "createShadowMapTexture" << std::endl;

  auto shadowMap = vars.reCreate<Texture>("csm.shadowMap",GL_TEXTURE_CUBE_MAP,GL_DEPTH_COMPONENT24,1,vars.getUint32("csm.resolution"),vars.getUint32("csm.resolution"));
  shadowMap->texParameteri(GL_TEXTURE_MIN_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_MAG_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_S      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_T      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL              );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_R_TO_TEXTURE);
}

void createShadowMapFBO(vars::Vars&vars){
  static Barrier barrier(vars,{"csm.shadowMap"});
  if(barrier.notChange())return;

  auto fbo = vars.reCreate<Framebuffer>("csv.shadowMapFBO");
  fbo->attachTexture(GL_DEPTH_ATTACHMENT,vars.get<Texture>("csm.shadowMap"));
}

void createShadowMapVAO(vars::Vars&vars){
  static Barrier barrier(vars,{"renderModel"});
  if(barrier.notChange())return;

  auto vao = vars.reCreate<VertexArray>("csv.shadowMapVAO");
  vao->addAttrib(vars.get<RenderModel>("renderModel")->vertices,0,3,GL_FLOAT);
}

void createShadowMapProgram(vars::Vars&vars){
  static Barrier barrier(vars);
  if(barrier.notChange())return;

#include<CubeShadowMappingShaders.h>
  vars.reCreate<Program>("csv.shadowMapProgram",
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMapVertexShaderSource  ),
      make_shared<Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        createShadowMapGeometryShaderSource));
}

void createMaskFBO(vars::Vars&vars){
  static Barrier barrier(vars,{"shadowMask"});
  if(barrier.notChange())return;

  auto fbo = vars.reCreate<Framebuffer>("csv.maskFBO");
  fbo->attachTexture(GL_COLOR_ATTACHMENT0,vars.get<Texture>("shadowMask"));
  fbo->drawBuffers(1,GL_COLOR_ATTACHMENT0);
}

void createShadowMaskProgram(vars::Vars&vars){
  static Barrier barrier(vars);
  if(barrier.notChange())return;

#include<CubeShadowMappingShaders.h>

  vars.reCreate<Program>("csv.shadowMaskProgram",
      make_shared<Shader>(GL_VERTEX_SHADER  ,
        "#version 450\n",
        createShadowMaskVertexShaderSource),
      make_shared<Shader>(GL_FRAGMENT_SHADER,
        "#version 450\n",
        createShadowMaskFragmentShaderSource));
}

void createShadowMaskVAO(vars::Vars&vars){
  static Barrier barrier(vars);
  if(barrier.notChange())return;

  vars.reCreate<VertexArray>("csv.maskVAO");
}

CubeShadowMapping::CubeShadowMapping(
        vars::Vars&vars       ):
  ShadowMethod(vars)
{

}

CubeShadowMapping::~CubeShadowMapping(){
  vars.erase("csm.shadowMap");
  vars.erase("csv.shadowMapFBO");
  vars.erase("csv.shadowMapVAO");
  vars.erase("csv.shadowMapProgram");

  vars.erase("csv.maskFBO");
  vars.erase("csv.maskVAO");
  vars.erase("csv.shadowMaskProgram");
}

void CubeShadowMapping::fillShadowMap(glm::vec4 const&lightPosition){
  createShadowMapTexture(vars);
  createShadowMapFBO(vars);
  createShadowMapVAO(vars);
  createShadowMapProgram(vars);

  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(2.5,10);
  auto const resolution = vars.getUint32("csm.resolution");
  auto const near       = vars.getFloat ("csm.near"      );
  auto const far        = vars.getFloat ("csm.far"       );
  auto const faces      = vars.getUint32("csm.faces"     );
  glViewport(0,0,resolution,resolution);
  glEnable(GL_DEPTH_TEST);
  vars.get<Framebuffer>("csv.shadowMapFBO")->bind();
  glClear(GL_DEPTH_BUFFER_BIT);
  vars.get<VertexArray>("csv.shadowMapVAO")->bind();
  vars.get<Program>("csv.shadowMapProgram")
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  glDrawArraysInstanced(GL_TRIANGLES,0,vars.get<RenderModel>("renderModel")->nofVertices,faces);
  vars.get<VertexArray>("csv.shadowMapVAO")->unbind();
  vars.get<Framebuffer>("csv.shadowMapFBO")->unbind();
}

void CubeShadowMapping::fillShadowMask(glm::vec4 const&lightPosition){
  createShadowMaskVAO(vars);
  createMaskFBO(vars);
  createShadowMaskProgram(vars);

  auto const near = vars.getFloat("csm.near");
  auto const far = vars.getFloat("csm.far");
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  glViewport(0,0,windowSize.x,windowSize.y);
  vars.get<Framebuffer>("csv.maskFBO")->bind();
  vars.get<VertexArray>("csv.maskVAO")->bind();
  vars.get<Program>("csv.shadowMaskProgram")
    ->set4fv("lightPosition",glm::value_ptr(lightPosition))
    ->set1f ("near"         ,near           )
    ->set1f ("far"          ,far            )
    ->use();
  vars.get<GBuffer>("gBuffer")->position->bind(0);
  vars.get<Texture>("csm.shadowMap")->bind(1);
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  vars.get<VertexArray>("csv.maskVAO")->unbind();
  vars.get<Framebuffer>("csv.maskFBO")->unbind();
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

