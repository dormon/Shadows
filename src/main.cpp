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
  std::shared_ptr<basicCamera::CameraTransform>  cameraTransform  = nullptr;
  std::shared_ptr<basicCamera::CameraProjection> cameraProjection = nullptr;
  std::shared_ptr<ShadowMethod>                  shadowMethod     = nullptr;

  CameraParam cameraParam;

  vars::Vars vars;

  virtual void                init() override;
  void                        parseArguments();
  void                        initWavefrontSize();
  void                        initCamera();
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
  *vars.add<glm::uvec2 >("windowSize"     ) = vector2uvec2(arg->getu32v("--window-size", {512, 512}, "window size"));
  *vars.add<glm::vec4  >("lightPosition"  ) = vector2vec4(arg->getf32v("--light", {0.f, 1000.f, 0.f, 1.f}, "light position"));
  vars.addString("modelName"      ) = arg->gets("--model", "/media/windata/ft/prace/models/2tri/2tri.3ds","model file name");
  vars.addBool  ("useShadows"     ) = !arg->isPresent("--no-shadows", "turns off shadows");
  vars.addBool  ("verbose"        ) = arg->isPresent("--verbose", "toggle verbose mode");
  vars.addString("methodName"     ) = arg->gets("--method", "","name of shadow method: ""cubeShadowMapping/cssv/sintorn/rssv/vssv/cssvsoe");
  vars.addSizeT ("wavefrontSize"  ) = arg->getu32("--wavefrontSize", 0,"warp/wavefront size, usually 32 for NVidia and 64 for AMD");
  vars.addSizeT ("maxMultiplicity") = arg->getu32("--maxMultiplicity", 2,"max number of triangles that share the same edge");
  vars.addBool  ("zfail"          ) = arg->getu32("--zfail", 1, "shadow volumes zfail 0/1");

  loadCubeShadowMappingParams(vars,arg);
  loadCSSVParams             (vars,arg);
  loadVSSVParams             (vars,arg);
  loadSintornParams          (vars,arg);
  loadRSSVParams             (vars,arg);
  loadTestParams             (vars,arg);
  cameraParam = CameraParam(arg);

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

void Shadows::initCamera() {
  assert(this != nullptr);
  cameraTransform  = createView(cameraParam);
  cameraProjection = createProjection(cameraParam, *vars.get<glm::uvec2>("windowSize"));
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
    cameraParam.type = "free";

  initCamera();

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
  if (vars.getBool("verbose") || shadowMethod || isTest)
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
  vars.get<RenderModel>("renderModel")->draw(cameraProjection->getProjection() *
                    cameraTransform->getView());
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
    auto keypoint =
        cameraPath->getKeypoint(float(k) / float(vars.getSizeT("test.flyLength")));
    auto flc =
        std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
    flc->setPosition(keypoint.position);
    flc->setRotation(keypoint.viewVector, keypoint.upVector);

    for (size_t f = 0; f < vars.getSizeT("test.framesPerMeasurement"); ++f) drawScene();

    std::vector<std::string> line;
    if (csv.size() == 0) {
      line.push_back("frame");
      for (auto const& x : measurement)
        if (x.first != "") line.push_back(x.first);
      csv.push_back(line);
      line.clear();
    }
    line.push_back(txtUtils::valueToString(k));
    for (auto const& x : measurement)
      if (x.first != "")
        line.push_back(txtUtils::valueToString(
            x.second / float(vars.getSizeT("test.framesPerMeasurement"))));
    csv.push_back(line);
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

  moveCameraWSAD(cameraParam, cameraTransform, keyDown);

  drawScene();

  //*
  if (vars.getString("methodName") == "sintorn") {
    auto sintorn = std::dynamic_pointer_cast<Sintorn>(shadowMethod);
    if (keyDown['h']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(sintorn->_HDT[0]);
    if (keyDown['j']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(sintorn->_HDT[1]);
    if (keyDown['k']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(sintorn->_HDT[2]);
    if (keyDown['l']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(sintorn->_HDT[3]);

    if (keyDown['v']) sintorn->drawHST(0);
    if (keyDown['b']) sintorn->drawHST(1);
    if (keyDown['n']) sintorn->drawHST(2);
    if (keyDown['m']) sintorn->drawHST(3);
    if (keyDown[',']) sintorn->drawFinalStencilMask();
  }
  if (vars.getString("methodName") == "rssv") {
    auto rssv = std::dynamic_pointer_cast<RSSV>(shadowMethod);
    if (keyDown['h']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(rssv->_HDT[0]);
    if (keyDown['j']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(rssv->_HDT[1]);
    if (keyDown['k']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(rssv->_HDT[2]);
    if (keyDown['l']) vars.get<DrawPrimitive>("drawPrimitive")->drawTexture(rssv->_HDT[3]);
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
  if (DOWN && event.key.keysym.sym == 'p') printCameraPosition(cameraTransform);
}

void Shadows::mouseMove(SDL_Event const& event) {
  mouseMoveCamera(cameraTransform, event, cameraParam,
                  *vars.get<glm::uvec2>("windowSize"));
}
