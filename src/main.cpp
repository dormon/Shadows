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

#include <Application.h>
#include <CameraParam.h>
#include <TestParam.h>
#include <mainUtil.h>
#include <util.h>

#include <ArgumentViewer/ArgumentViewer.h>
#include <BasicCamera/FreeLookCamera.h>
#include <BasicCamera/OrbitCamera.h>
#include <BasicCamera/PerspectiveCamera.h>
#include <CSSV.h>
#include <CSSVSOE.h>
#include <CSV.h>
#include <CameraPath.h>
#include <CubeShadowMapping.h>
#include <CubeSMParam.h>
#include <Deferred.h>
#include <DrawPrimitive.h>
#include <Model.h>
#include <RSSV.h>
#include <RSSVTiles.h>
#include <Shading.h>
#include <ShadowMethod.h>
#include <Sintorn.h>
#include <SintornParam.h>
#include <SintornTiles.h>
#include <TimeStamp.h>
#include <VSSV.h>
#include <VSSVParam.h>
#include <Vars.h>

class Shadows : public simple3DApp::Application {
 public:
  Shadows(int argc, char* argv[]) : Application(argc, argv) {}
  virtual void draw() override;
  std::shared_ptr<ShadowMethod>                  shadowMethod     = nullptr;

  vars::Vars vars;

  virtual void                init() override;
  void                        parseArguments();
  void                        initWavefrontSize();
  void                        measure();
  void                        drawScene();
  virtual void                mouseMove(SDL_Event const& event) override;
  std::map<SDL_Keycode, bool> keyDown;
  virtual void                key(SDL_Event const& e, bool down) override;
  void ifExistStamp(std::string const&n);
  void ifExistBeginStamp();
  void ifExistEndStamp(std::string const&n);
};

void Shadows::parseArguments() {
  assert(this != nullptr);
  auto arg = std::make_shared<argumentViewer::ArgumentViewer>(argc, argv);
  loadBasicApplicationParameters(vars,arg);
  loadCubeShadowMappingParams(vars,arg);
  loadCSSVParams             (vars,arg);
  loadVSSVParams             (vars,arg);
  loadSintornParams          (vars,arg);
  loadRSSVParams             (vars,arg);
  loadTestParams             (vars,arg);
  loadCameraParams           (vars,arg);

  vars.addSizeT("cssvsoe.computeSidesWGS") = arg->getu32(
      "--cssvsoe-WGS", 64, "compute silhouette shadow volumes work group size");

  bool printHelp = arg->isPresent("-h", "prints this help");
  if (printHelp || !arg->validate()) {
    std::cerr << arg->toStr();
    exit(0);
  }
}

void Shadows::initWavefrontSize() {
  vars.getSizeT("wavefrontSize") = getWavefrontSize(vars.getSizeT("wavefrontSize"));
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
    vars.getString("camera.type") = "free";

  createView      (vars);
  createProjection(vars);

  vars.add<GBuffer        >("gBuffer"    ,windowSize.x, windowSize.y);
  vars.add<Model          >("model"      ,*vars.get<std::string>("modelName"));
  vars.add<RenderModel    >("renderModel",vars.get<Model>("model"));
  vars.add<ge::gl::Texture>("shadowMask" ,(GLenum)GL_TEXTURE_2D,(GLenum)GL_R32F, 1,(GLsizei)windowSize.x,(GLsizei)windowSize.y);
  vars.add<Shading        >("shading"    ,vars);

  if      (vars.getString("methodName") == "cubeShadowMapping")shadowMethod = std::make_shared<CubeShadowMapping>(vars);
  else if (vars.getString("methodName") == "cssv"             )shadowMethod = std::make_shared<CSSV>(vars);
  else if (vars.getString("methodName") == "cssvsoe"          )shadowMethod = std::make_shared<CSSVSOE>(vars);
  else if (vars.getString("methodName") == "sintorn"          )shadowMethod = std::make_shared<Sintorn>(vars);
  else if (vars.getString("methodName") == "rssv"             )shadowMethod = std::make_shared<RSSV>(vars);
  else if (vars.getString("methodName") == "vssv"             )shadowMethod = std::make_shared<VSSV>(vars);
  else vars.getBool("useShadows") = false;

  bool isTest = vars.getString("test.name") == "fly";
  if (vars.getBool("verbose") || (shadowMethod && isTest))
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
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  ge::gl::glViewport(0, 0, windowSize.x, windowSize.y);
  ge::gl::glEnable(GL_DEPTH_TEST);
  vars.get<GBuffer>("gBuffer")->begin();
  ge::gl::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
                  GL_STENCIL_BUFFER_BIT);
  vars.get<ge::gl::Texture>("shadowMask")->clear(0, GL_RED, GL_FLOAT);
  auto const cameraProjection = vars.getReinterpret<basicCamera::CameraProjection>("cameraProjection");
  auto const cameraTransform  = vars.getReinterpret<basicCamera::CameraTransform >("cameraTransform" );
  vars.get<RenderModel>("renderModel")->draw(cameraProjection->getProjection() * cameraTransform->getView());
  vars.get<GBuffer>("gBuffer")->end();

  ifExistStamp("gBuffer");

  if (shadowMethod)
    shadowMethod->create(*vars.get<glm::vec4>("lightPosition"),
                         cameraTransform->getView(),
                         cameraProjection->getProjection());

  ge::gl::glDisable(GL_DEPTH_TEST);
  vars.get<Shading>("shading")->draw(*vars.get<glm::vec4>("lightPosition"),
                glm::vec3(glm::inverse(cameraTransform->getView()) *
                          glm::vec4(0, 0, 0, 1)),
                *vars.get<bool>("useShadows"));
  ifExistEndStamp("shading");
}

void writeCSVHeaderIfFirstLine(
    std::vector<std::vector<std::string>>&csv,
    std::map<std::string,float>const&measurement){
  if (csv.size() != 0) return;
  std::vector<std::string>line;
  line.push_back("frame");
  for (auto const& x : measurement)
    if (x.first != "") line.push_back(x.first);
  csv.push_back(line);
}

void writeMeasurementIntoCSV(
    vars::Vars&vars,
    std::vector<std::vector<std::string>>&csv,
    std::map<std::string,float>const&measurement,
    size_t idOfMeasurement){
  std::vector<std::string> line;
  line.push_back(txtUtils::valueToString(idOfMeasurement));
  for (auto const& x : measurement)
    if (x.first != "")
      line.push_back(txtUtils::valueToString(
          x.second / float(vars.getSizeT("test.framesPerMeasurement"))));
  csv.push_back(line);
}

void setCameraAccordingToKeyFrame(std::shared_ptr<CameraPath>const&cameraPath,vars::Vars&vars,size_t keyFrame){
    auto keypoint =
        cameraPath->getKeypoint(float(keyFrame) / float(vars.getSizeT("test.flyLength")));
    auto flc = vars.getReinterpret<basicCamera::FreeLookCamera>("cameraTransform");
    flc->setPosition(keypoint.position);
    flc->setRotation(keypoint.viewVector, keypoint.upVector);
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
  assert(this != nullptr);

  if (vars.getString("test.name") == "fly") {
    measure();
    return;
  }

  moveCameraWSAD(vars, keyDown);

  drawScene();

  //*
  if (vars.getString("methodName") == "sintorn") {
    auto sintorn = std::dynamic_pointer_cast<Sintorn>(shadowMethod);
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
    auto rssv = std::dynamic_pointer_cast<RSSV>(shadowMethod);
    auto dp = vars.get<DrawPrimitive>("drawPrimitive");
    auto drawTex = [&](char s,int i){if (keyDown[s]) dp->drawTexture(rssv->_HDT[i]);};
    for(int i=0;i<4;++i)drawTex("hjkl"[i],i);
  }

  // */
  swap();
}

void testRSSVTiler(size_t w, size_t h, size_t warp) {
  auto win = glm::uvec2(w, h);
  std::cout << "window: " << win.x << "x" << win.y << "warp: " << warp
            << std::endl;
  auto d = rssvGetMaxUpperTileDivisibility(win, warp);
  for (auto const& x : d) std::cout << x.x << "x" << x.y << std::endl;
  auto dd = d;
  for (size_t i = 1; i < dd.size(); ++i)
    dd[dd.size() - 1 - i] *= dd[dd.size() - i];
  for (auto const& x : dd)
    std::cout << "pix: " << x.x << "x" << x.y << std::endl;
  std::cout << std::endl;
}

void printTiling(size_t w, size_t h, size_t t) {
  RSSVTilingSizes tiling(glm::uvec2(w, h), t);
  std::cout << w << " x " << h << " : " << t << std::endl;
  for (size_t i = 0; i < tiling.borderTileDivisibilityIntoTiles.size(); ++i) {
    std::cout << "FULL___TILE_DIVISIBILITY_INTO_PIXELS    ("
              << uvec2ToStr(tiling.full__TileDivisibilityIntoPixels.at(i))
              << ")" << std::endl;
    std::cout << "FULL___TILE_DIVISIBILITY_INTO_TILES     ("
              << uvec2ToStr(tiling.full__TileDivisibilityIntoTiles.at(i)) << ")"
              << std::endl;
    std::cout << "BORDER_TILE_DIVISIBILITY_INTO_PIXELS    ("
              << uvec2ToStr(tiling.borderTileDivisibilityIntoPixels.at(i))
              << ")" << std::endl;
    std::cout << "BORDER_TILE_DIVISIBILITY_INTO_TILES     ("
              << uvec2ToStr(tiling.borderTileDivisibilityIntoTiles.at(i)) << ")"
              << std::endl;
    std::cout << "BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES("
              << uvec2ToStr(tiling.borderTileDivisibilityIntoFullTiles.at(i))
              << ")" << std::endl;
    std::cout << "HDT_SIZE                                ("
              << uvec2ToStr(tiling.hdtSize.at(i)) << ")" << std::endl;
    std::cout << std::endl;
  }
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

void Shadows::mouseMove(SDL_Event const& event) {
  mouseMoveCamera(vars, event);
}
