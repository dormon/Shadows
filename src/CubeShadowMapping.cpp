#include<CubeShadowMapping.h>
#include<Deferred.h>
#include<Model.h>

using namespace ge::gl;
using namespace std;

#include<Vars/Resource.h>
class Barrier{
  public:
    Barrier(vars::Vars&vars,std::vector<std::string>const&inputs = {}):vars(vars){
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

std::string getFullMethodName(std::string const&cls,std::string const&method){
  return cls + "._methods." + method;
}

std::string getMethodTableName(std::string const&cls){
  return cls + "._methodNames";
}

void addMethodTable(vars::Vars&vars,std::string const&cls){
  auto const methodTableName = getMethodTableName(cls);
  if(!vars.has(methodTableName))
    vars.add<std::set<std::string>>(methodTableName);
}

void registerMethod(vars::Vars&vars,std::string const&cls,std::string const&method){
  auto const methodTableName = getMethodTableName(cls);
  auto const fullMethodName  = getFullMethodName(cls,method);

  addMethodTable(vars,cls);

  auto methodTable = vars.get<std::set<std::string>>(methodTableName);
  methodTable->insert(fullMethodName);
}

Barrier*createOrGetBarrier(
    vars::Vars                   &vars  ,
    std::string             const&cls   ,
    std::string             const&method,
    std::vector<std::string>const&v     ){
  auto const fullName = getFullMethodName(cls,method);
  if(!vars.has(fullName)){
    registerMethod(vars,cls,method);
    return vars.add<Barrier>(fullName,vars,v);
  }
  return vars.get<Barrier>(fullName);
}

void callDestructor(vars::Vars&vars,std::string const&cls){
  auto const methodTableName = getMethodTableName(cls);
  if(!vars.has(methodTableName))return;
  auto methodNames = vars.get<std::set<std::string>>(methodTableName);
  for(auto const&method:*methodNames){
    auto const fullName = getFullMethodName(cls,method);
    vars.erase(fullName);
  }
  vars.erase(methodTableName);
}

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)

void createShadowMapTexture(vars::Vars&vars){
  auto barrier = createOrGetBarrier(vars,"csm",STRINGIZE(__FUNCTION__),{"csm.resolution"});
  if(barrier->notChange())return;

  auto const resolution = vars.getUint32("csm.resolution");
  auto shadowMap = vars.reCreate<Texture>("csm.shadowMap",GL_TEXTURE_CUBE_MAP,GL_DEPTH_COMPONENT24,1,resolution,resolution);
  shadowMap->texParameteri(GL_TEXTURE_MIN_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_MAG_FILTER  ,GL_NEAREST             );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_S      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_WRAP_T      ,GL_CLAMP_TO_EDGE       );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_FUNC,GL_LEQUAL              );
  shadowMap->texParameteri(GL_TEXTURE_COMPARE_MODE,GL_COMPARE_R_TO_TEXTURE);
}

void createShadowMapFBO(vars::Vars&vars){
  auto barrier = createOrGetBarrier(vars,"csm",STRINGIZE(__FUNCTION__),{"csm.shadowMap"});
  if(barrier->notChange())return;

  auto fbo = vars.reCreate<Framebuffer>("csv.shadowMapFBO");
  fbo->attachTexture(GL_DEPTH_ATTACHMENT,vars.get<Texture>("csm.shadowMap"));
}


void createShadowMapVAO(vars::Vars&vars){
  auto barrier = createOrGetBarrier(vars,"csm",STRINGIZE(__FUNCTION__),{"csm.shadowMap"});
  if(barrier->notChange())return;

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
  callDestructor(vars,"csm");

  vars.erase("csm.shadowMap"       );
  vars.erase("csm.shadowMapFBO"    );
  vars.erase("csm.shadowMapVAO"    );
  vars.erase("csm.shadowMapProgram");

  vars.erase("csm.maskFBO"          );
  vars.erase("csm.maskVAO"          );
  vars.erase("csm.shadowMaskProgram");
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

