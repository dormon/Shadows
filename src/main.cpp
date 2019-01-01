#include <SDL2CPP/Window.h>
#include <TxtUtils/TxtUtils.h>
#include <geGL/OpenGLContext.h>
#include <geGL/StaticCalls.h>
#include <geGL/VertexArray.h>
#include <geGL/geGL.h>
#include <cmath>
#include <limits>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include<FreeImagePlus.h>

#include <Simple3DApp/Application.h>
#include <CameraParam.h>
#include <TestParam.h>
#include <mainUtil.h>
#include <util.h>

#include <ArgumentViewer/ArgumentViewer.h>
#include <BasicCamera/FreeLookCamera.h>
#include <BasicCamera/OrbitCamera.h>
#include <BasicCamera/PerspectiveCamera.h>
#include <CSSV/CSSV.h>
#include <CSSVSOE.h>
#include <CSV.h>
#include <CameraPath.h>
#include <CubeShadowMapping/CubeShadowMapping.h>
#include <CubeShadowMapping/Params.h>
#include <Deferred.h>
#include <DrawPrimitive.h>
#include <Model.h>
#include <RSSV/RSSV.h>
#include <RSSV/Tiles.h>
#include <Shading.h>
#include <ShadowMethod.h>
#include <Sintorn/Sintorn.h>
#include <Sintorn/Param.h>
#include <TimeStamp.h>
#include <VSSV/VSSV.h>
#include <VSSV/Params.h>
#include <RayTracing/RayTracing.h>
#include <Vars/Vars.h>
#include <Vars/Caller.h>
#include <createGBuffer.h>

#include <imguiVars.h>

#include <FunctionPrologue.h>
#include <Methods.h>
#include <geGL/OpenGLUtil.h>

#define ___ std::cerr << __FILE__ << ": " << __LINE__ << std::endl

class Shadows : public simple3DApp::Application {
 public:
  Shadows(int argc, char* argv[]) : Application(argc, argv) {}
  virtual void draw() override;

  vars::Vars vars;

  virtual void                init() override;
  void                        parseArguments();
  void                        initWavefrontSize();
  void                        measure();
  void                        drawScene();
  virtual void                mouseMove(SDL_Event const& event) override;
  std::map<SDL_Keycode, bool> keyDown;
  virtual void                key(SDL_Event const& e, bool down) override;
  virtual void                resize(uint32_t x,uint32_t y) override;
  void ifExistStamp(std::string const&n);
  void ifExistBeginStamp();
  void ifExistEndStamp(std::string const&n);
};

void Shadows::parseArguments() {
  assert(this != nullptr);

  auto methods = vars.add<Methods>("methods");
  methods->add<CubeShadowMapping>("cubeShadowMapping");
  methods->add<cssv::CSSV       >("cssv"             );
  methods->add<CSSVSOE          >("cssvsoe"          );
  methods->add<Sintorn          >("sintorn"          );
  methods->add<rssv::RSSV       >("rssv"             );
  methods->add<VSSV             >("vssv"             );
  methods->add<RayTracing       >("rayTracing"       );

  auto arg = std::make_shared<argumentViewer::ArgumentViewer>(argc, argv);
  loadBasicApplicationParameters(vars,arg);
  loadCubeShadowMappingParams   (vars,arg);
  cssv::loadParams              (vars,arg);
  loadVSSVParams                (vars,arg);
  loadSintornParams             (vars,arg);
  rssv::loadParams              (vars,arg);
  loadTestParams                (vars,arg);
  loadCameraParams              (vars,arg);

  vars.addSizeT("cssvsoe.computeSidesWGS") = arg->getu32(
      "--cssvsoe-WGS", 64, "compute silhouette shadow volumes work group size");

  vars.addSizeT("frameCounter");
  vars.addSizeT("maxFrame") = arg->getu32("--maxFrame",0,"after this frame the app will stop");


  bool printHelp = arg->isPresent("-h", "prints this help");
  if (printHelp || !arg->validate()) {
    std::cerr << arg->toStr();
    exit(0);
  }
}

void Shadows::initWavefrontSize() {
  FUNCTION_CALLER();
  vars.getSizeT("wavefrontSize") = getWavefrontSize(vars.getSizeT("wavefrontSize"));
}

void createGeometryBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","windowSize");

  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  vars.reCreate<GBuffer>("gBuffer",windowSize.x, windowSize.y);
}

void createShadowMask(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","windowSize");

  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  vars.reCreate<ge::gl::Texture>("shadowMask" ,(GLenum)GL_TEXTURE_2D,(GLenum)GL_R32F, 1,(GLsizei)windowSize.x,(GLsizei)windowSize.y);
}

void createMethod(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","methodName");

  auto const methodName = vars.getString("methodName"); 
  auto methods = vars.get<Methods>("methods");
  methods->createMethod(methodName,vars);
}

void testComputeShader(){
  auto cs = std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER,
  R".(
  #version 450
  layout(local_size_x=64)in;

  layout(binding=0,std430)buffer Data{uint data[];};

  void main(){
    data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x;
  }
  ).");
  auto prg = std::make_shared<ge::gl::Program>(cs);
  prg->use();
  auto data = std::vector<uint32_t>(64*2);
  for(auto&x:data)
    x = 10;
  auto buf = std::make_shared<ge::gl::Buffer>(data);
  buf->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  ge::gl::glDispatchCompute(2,1,1);
  ge::gl::glFinish();
  buf->getData(data.data());

}

void Shadows::init() {
  FUNCTION_CALLER();

  testComputeShader();

  parseArguments();

  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  window->setSize(windowSize.x, windowSize.y);

  ge::gl::glEnable(GL_DEPTH_TEST);
  ge::gl::glDepthFunc(GL_LEQUAL);
  ge::gl::glDisable(GL_CULL_FACE);
  ge::gl::glClearColor(0, 0, 0, 1);

  initWavefrontSize();

  if (vars.getString("test.name") == "fly" || vars.getString("test.name") == "grid")
    vars.getString("args.camera.type") = "free";


  createView      (vars);
  createProjection(vars);

  createGeometryBuffer(vars);
  createShadowMask(vars);
  vars.add<Model          >("model"      ,vars.getString("modelName"));
  vars.add<RenderModel    >("renderModel",vars.get<Model>("model"));
  vars.add<Shading        >("shading"    ,vars);

  createMethod(vars);

  bool isTest = vars.getString("test.name") == "fly";
  if (vars.getBool("verbose") || (vars.has("shadowMethod") && isTest))
    vars.add<TimeStamp>("timeStamp");

  vars.add<DrawPrimitive>("drawPrimitive",windowSize);

}

void Shadows::ifExistStamp(std::string const&n){
  if (vars.has("timeStamp")) vars.get<TimeStamp>("timeStamp")->stamp(n);
}

void Shadows::ifExistBeginStamp(){
  if (vars.has("timeStamp")) vars.get<TimeStamp>("timeStamp")->begin();
}

void Shadows::ifExistEndStamp(std::string const&n){
  if (vars.has("timeStamp")) vars.get<TimeStamp>("timeStamp")->end(n);
}

void Shadows::drawScene() {
  //createMethod(vars);

  ifExistBeginStamp();

  createGBuffer(vars);

  ifExistStamp("gBuffer");

  ifMethodExistCreateShadowMask(vars);

  doShading(vars);

  ifExistEndStamp("shading");
}


void Shadows::measure() {
  FUNCTION_CALLER();

  assert(this != nullptr);
  if (vars.getString("test.flyKeyFileName") == "") {
    std::cerr << "camera path file is empty" << std::endl;
    mainLoop->removeWindow(window->getId());
    return;
  }
  auto cameraPath =
      std::make_shared<CameraPath>(false, vars.getString("test.flyKeyFileName"));
  std::map<std::string, float> measurement;
  vars.get<TimeStamp>("timeStamp")->setPrinter([&](std::vector<std::string> const& names,
                              std::vector<float> const&       values) {
    for (size_t i = 0; i < names.size(); ++i)
      if (names[i] != "") {
        if (measurement.count(names[i]) == 0) measurement[names[i]] = 0.f;
        measurement[names[i]] += values[i];
      }
  });

  std::vector<std::vector<std::string>> csv;
  for (size_t k = 0; k < vars.getSizeT("test.flyLength"); ++k) {
    setCameraAccordingToKeyFrame(cameraPath,vars,k);

    for (size_t f = 0; f < vars.getSizeT("test.framesPerMeasurement"); ++f) drawScene();

    writeCSVHeaderIfFirstLine(csv,measurement);
    writeMeasurementIntoCSV(vars,csv,measurement,k);

    measurement.clear();
    window->swap();
  }
  std::string output = vars.getString("test.outputName") + ".csv";
  saveCSV(output, csv);
  mainLoop->removeWindow(window->getId());
}

void selectMethod(vars::Vars&vars){
  auto methods = vars.get<Methods>("methods");
  auto method = vars.getString("methodName");
  int oldMethodId;
  if(methods->hasMethod(method))
    oldMethodId = methods->getId(vars.getString("methodName"));
  else
    oldMethodId = methods->getNofMethods();
  int newMethodId = oldMethodId;

  std::vector<char const*>names;
  for(size_t i=0;i<methods->getNofMethods();++i)
    names.push_back(methods->getName(i).c_str());
  names.push_back("no shadow");
  
  ImGui::ListBox("method",&newMethodId,names.data(),names.size());
  if(newMethodId != oldMethodId){
    if(newMethodId < methods->getNofMethods())
      vars.getString("methodName") = methods->getName(newMethodId);
    else
      vars.getString("methodName") = "no shadow";
    vars.updateTicks("methodName");
  }
}

size_t basicTexelSize(ge::gl::BasicInternalFormatElement const&info){
  size_t result = 0;
  for(auto const&x:info.channelSize)
    result += x;
  return result;
}

void saveBasicTexture(std::string const&name,ge::gl::Texture const*tex){
  auto const iFormat   = tex->getInternalFormat(0);
  auto const info      = ge::gl::getBasicInternalFormatInformation(iFormat);
  auto const width     = tex->getWidth(0);
  auto const height    = tex->getHeight(0);
  auto const texelSize = basicTexelSize(info);
  auto const nofTexels = width * height;
  auto buffer = std::vector<uint8_t>(texelSize * nofTexels);
  if(info.type == ge::gl::BasicInternalFormatElement::UNSIGNED_INT){
    //ge::gl::glGetTextureImage(); tex,level,format,type,bufSize,pixels
    //
    //ge::gl::glGetTextureSubImage(); tex,level,xoff,yoff,zoff,w,h,d,format,type,bufSize,pixels
  }
}

void saveDepthTexture(std::string const&name,ge::gl::Texture const*tex){
}

void saveCompressedTexture(std::string const&name,ge::gl::Texture const*tex){

}

void saveTexture(std::string const&name,ge::gl::Texture const*tex){
  auto const iFormat   = tex->getInternalFormat(0);
  if(ge::gl::isInternalFormatBasic(iFormat)){
    saveBasicTexture(name,tex);
    return;
  }
  if(ge::gl::isInternalFormatDepth(iFormat)){
    saveDepthTexture(name,tex);
    return;
  }
  if(ge::gl::isInternalFormatDepth(iFormat)){
    saveCompressedTexture(name,tex);
    return;
  }
/*

  auto const info      = ge::gl::getBasicInternalFormatInformation(iFormat);
  auto const nofChannels = ge::gl::nofInternalFormatChannels(iFormat);
  size_t const channelSizes[4] = {
    ge::gl::internalFormatChannelSize(iFormat,0),
    ge::gl::internalFormatChannelSize(iFormat,1),
    ge::gl::internalFormatChannelSize(iFormat,2),
    ge::gl::internalFormatChannelSize(iFormat,3),
  };
  auto const floatingPoint = ge::gl::internalFormatFloatingPoint(iFormat);
  auto const signedType    = ge::gl::internalFormatSigned(iFormat);
  auto const fixedPoint    = ge::gl::internalFormatFixedPoint(iFormat);
  auto const texelSize     = ge::gl::internalFormatSize(iFormat);
  fipImage img;
  std::vector<uint8_t>buf(nofTexels * texelSize);

  ge::gl::glGetTextureImage(id,0,GL_RED,GL_FLOAT,buf.size(),buf.data());
  img.setSize(FIT_FLOAT,width,height,32);
  for(size_t y=0;y<height;++y){
    auto ptr = (float*)FreeImage_GetScanLine(img,y);
    for(size_t x=0;x<width;++x)
      ptr[x] = buf.at(y*width + x);
  }
  img.save(name.c_str());
  */
}

void Shadows::draw() {
  FUNCTION_CALLER();

  //ge::gl::glClear(GL_COLOR_BUFFER_BIT);
#if 1
  createGeometryBuffer(vars);
  createShadowMask(vars);
  createProjection(vars);
  createMethod(vars);

  if(vars.getSizeT("maxFrame") != 0){
    if(vars.getSizeT("frameCounter") >= vars.getSizeT("maxFrame"))
      mainLoop->removeWindow(window->getId());
    vars.getSizeT("frameCounter")++;
  }

  ge::gl::glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

#if 1
  assert(this != nullptr);

  if (vars.getString("test.name") == "fly") {
    measure();
    return;
  }

  moveCameraWSAD(vars, keyDown);

  drawScene();

  //*
  //std::cerr << vars.getString("methodName") << std::endl;
  if (vars.getString("methodName") == "sintorn") {
    //std::cerr << "asd" << std::endl;
    auto sintorn = vars.getReinterpret<Sintorn>("shadowMethod");
    auto dp = vars.get<DrawPrimitive>("drawPrimitive");
    auto drawTex = [&](char s,int i){if (keyDown[s]) dp->drawTexture(vars.getVector<std::shared_ptr<ge::gl::Texture>>("sintorn.HDT")[i]);};
    for(int i=0;i<4;++i)drawTex("hjkl"[i],i);
    if (keyDown['v']) sintorn->drawHST(0);
    if (keyDown['b']) sintorn->drawHST(1);
    if (keyDown['n']) sintorn->drawHST(2);
    if (keyDown['m']) sintorn->drawHST(3);
    if (keyDown[',']) sintorn->drawFinalStencilMask();
  }
  if (vars.getString("methodName") == "rssv") {
    auto rssv = vars.getReinterpret<rssv::RSSV>("shadowMethod");
    auto dp = vars.get<DrawPrimitive>("drawPrimitive");
    auto drawTex = [&](char s,int i){if (keyDown[s]) dp->drawTexture(rssv->_HDT[i]);};
    for(int i=0;i<4;++i)drawTex("hjkl"[i],i);
  }
#endif


  //ImGui::BeginCombo("cici","was");
  //ImGui::EndCombo();

  //TODO imgui gui
  drawImguiVars(vars);



  selectMethod(vars);




  if(ImGui::Button("screenshot"))
  {
    fipImage img;
    auto windowSize = *vars.get<glm::uvec2>("windowSize");
    //img.setSize(FIT_BITMAP,->x,vars.get<glm::uvec2>("windowSize")->y,24);
    auto id = vars.get<ge::gl::Texture>("shadowMask")->getId();
    std::vector<float>buf(windowSize.x * windowSize.y);
    ge::gl::glGetTextureImage(id,0,GL_RED,GL_FLOAT,buf.size()*sizeof(float),buf.data());
    img.setSize(FIT_FLOAT,windowSize.x,windowSize.y,32);
    for(size_t y=0;y<windowSize.y;++y){
      auto ptr = (float*)FreeImage_GetScanLine(img,y);
      for(size_t x=0;x<windowSize.x;++x)
        ptr[x] = buf.at(y*windowSize.x + x);
    }
    img.save("/home/dormon/Desktop/test.exr");

    id = vars.get<GBuffer>("gBuffer")->color->getId();
    std::vector<uint8_t>buf1(windowSize.x * windowSize.y * sizeof(uint16_t) * 4);
    ge::gl::glGetTextureImage(id,0,GL_RGBA_INTEGER,GL_UNSIGNED_SHORT,buf1.size(),buf1.data());
    img.setSize(FIT_BITMAP,windowSize.x,windowSize.y,24);
    for(size_t y=0;y<windowSize.y;++y){
      auto ptr = (uint8_t*)FreeImage_GetScanLine(img,y);
      for(size_t x=0;x<windowSize.x;++x){
        ptr[x*3+0] = buf1.at((y*windowSize.x + x)*(sizeof(uint16_t)*4) + 0*sizeof(uint16_t));
        ptr[x*3+1] = buf1.at((y*windowSize.x + x)*(sizeof(uint16_t)*4) + 1*sizeof(uint16_t));
        ptr[x*3+2] = buf1.at((y*windowSize.x + x)*(sizeof(uint16_t)*4) + 2*sizeof(uint16_t));
      }
    }
    img.save("/home/dormon/Desktop/aa.png");


    
    std::cerr << "take a screenshot" << std::endl;
  }
#endif

  //ge::gl::glClearColor(0,0.4,0,1);
  //ge::gl::glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  // */
  swap();
}

int main(int argc, char* argv[]) {
  Shadows app{argc, argv};
  app.start();
  return EXIT_SUCCESS;
}

void Shadows::key(SDL_Event const& event, bool DOWN) {
  keyDown[event.key.keysym.sym] = DOWN;
  if (DOWN && event.key.keysym.sym == 'p') printCameraPosition(vars);
}

void Shadows::resize(uint32_t x,uint32_t y){
  FUNCTION_CALLER();

  auto windowSize = vars.get<glm::uvec2>("windowSize");
  windowSize->x = x;
  windowSize->y = y;
  vars.updateTicks("windowSize");
  ge::gl::glViewport(0,0,x,y);
}

void Shadows::mouseMove(SDL_Event const& event) {
  mouseMoveCamera(vars, event);
}

