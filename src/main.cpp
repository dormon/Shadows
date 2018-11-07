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
#include <createGBuffer.h>

#include <imguiVars.h>

#include<Barrier.h>

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
  vars.getSizeT("wavefrontSize") = getWavefrontSize(vars.getSizeT("wavefrontSize"));
}

void createGeometryBuffer(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{"windowSize"}))return;

  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  vars.reCreate<GBuffer>("gBuffer",windowSize.x, windowSize.y);
}

void createShadowMask(vars::Vars&vars){
  if(notChanged(vars,"all",__FUNCTION__,{"windowSize"}))return;

  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  vars.reCreate<ge::gl::Texture>("shadowMask" ,(GLenum)GL_TEXTURE_2D,(GLenum)GL_R32F, 1,(GLsizei)windowSize.x,(GLsizei)windowSize.y);
}

void Shadows::init() {
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

  if      (vars.getString("methodName") == "cubeShadowMapping")vars.add<CubeShadowMapping>("shadowMethod",vars);
  else if (vars.getString("methodName") == "cssv"             )vars.add<cssv::CSSV       >("shadowMethod",vars);
  else if (vars.getString("methodName") == "cssvsoe"          )vars.add<CSSVSOE          >("shadowMethod",vars);
  else if (vars.getString("methodName") == "sintorn"          )vars.add<Sintorn          >("shadowMethod",vars);
  else if (vars.getString("methodName") == "rssv"             )vars.add<rssv::RSSV       >("shadowMethod",vars);
  else if (vars.getString("methodName") == "vssv"             )vars.add<VSSV             >("shadowMethod",vars);
  else if (vars.getString("methodName") == "rayTracing"       )vars.add<RayTracing       >("shadowMethod",vars);
  else vars.getBool("useShadows") = false;

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
  ifExistBeginStamp();

  createGBuffer(vars);

  ifExistStamp("gBuffer");

  ifMethodExistCreateShadowMask(vars);

  doShading(vars);

  ifExistEndStamp("shading");
}


void Shadows::measure() {
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

void Shadows::draw() {
  createGeometryBuffer(vars);
  createShadowMask(vars);
  createProjection(vars);

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
  if (vars.getString("methodName") == "sintorn") {
    auto sintorn = vars.getReinterpret<Sintorn>("shadowMethod");
    auto dp = vars.get<DrawPrimitive>("drawPrimitive");
    auto drawTex = [&](char s,int i){if (keyDown[s]) dp->drawTexture(sintorn->_HDT[i]);};
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

  //TODO imgui gui
  drawImguiVars(vars);

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
    std::cerr << "take a screenshot" << std::endl;
  }



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
  auto windowSize = vars.get<glm::uvec2>("windowSize");
  windowSize->x = x;
  windowSize->y = y;
  vars.updateTicks("windowSize");
  ge::gl::glViewport(0,0,x,y);
}

void Shadows::mouseMove(SDL_Event const& event) {
  mouseMoveCamera(vars, event);
}

