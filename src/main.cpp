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

#include <ArgumentViewer/ArgumentViewer.h>
#include <BasicCamera/FreeLookCamera.h>
#include <BasicCamera/OrbitCamera.h>
#include <BasicCamera/PerspectiveCamera.h>
#include <CSSV.h>
#include <CSSVSOE.h>
#include <CSV.h>
#include <CameraPath.h>
#include <CubeShadowMapping.h>
#include <Deferred.h>
#include <DrawPrimitive.h>
#include <Model.h>
#include <RSSV.h>
#include <RSSVTiles.h>
#include <Shading.h>
#include <ShadowMethod.h>
#include <Sintorn.h>
#include <SintornTiles.h>
#include <TimeStamp.h>
#include <VSSV.h>

class Shadows : public simple3DApp::Application {
 public:
  Shadows(int argc, char* argv[]) : Application(argc, argv) {}
  virtual void                                   draw() override;
  std::shared_ptr<ge::gl::VertexArray>           emptyVAO         = nullptr;
  std::shared_ptr<GBuffer>                       gBuffer          = nullptr;
  std::shared_ptr<Model>                         model            = nullptr;
  std::shared_ptr<RenderModel>                   renderModel      = nullptr;
  std::shared_ptr<basicCamera::CameraTransform>  cameraTransform  = nullptr;
  std::shared_ptr<basicCamera::CameraProjection> cameraProjection = nullptr;
  std::shared_ptr<Shading>                       shading          = nullptr;
  std::shared_ptr<ge::gl::Texture>               shadowMask       = nullptr;
  std::shared_ptr<ShadowMethod>                  shadowMethod     = nullptr;
  std::shared_ptr<DrawPrimitive>                 drawPrimitive    = nullptr;
  std::shared_ptr<TimeStamp>                     timeStamper      = nullptr;
  glm::uvec2 windowSize = glm::uvec2(512u, 512u);

  std::string cameraType      = "orbit";
  float       cameraFovy      = glm::radians(90.f);
  float       cameraNear      = 0.1f;
  float       cameraFar       = 1000.f;
  float       sensitivity     = 0.01f;
  float       orbitZoomSpeed  = 0.2f;
  float       freeCameraSpeed = 1.f;

  glm::vec4 lightPosition = glm::vec4(100.f, 100.f, 100.f, 1.f);

  size_t wavefrontSize = 0;

  size_t maxMultiplicity = 2;

  ShadowVolumesParams     svParams;
  CubeShadowMappingParams cubeSMParams;
  CSSVParams              cssvParams;
  CSSVSOEParams           cssvsoeParams;
  VSSVParams              vssvParams;
  SintornParams           sintornParams;
  RSSVParams              rssvParams;

  std::string testName                 = "";
  std::string testFlyKeyFileName       = "";
  size_t      testFlyLength            = 0;
  size_t      testFramesPerMeasurement = 5;
  std::string testOutputName           = "measurement";

  std::string modelName  = "";
  std::string methodName = "";
  bool        verbose    = false;
  bool        useShadows = true;

  virtual void                init() override;
  void                        parseArguments();
  void                        initWavefrontSize();
  void                        initCamera();
  void                        measure();
  void                        drawScene();
  virtual void                mouseMove(SDL_Event const& event) override;
  std::map<SDL_Keycode, bool> keyDown;
  virtual void                key(SDL_Event const& e, bool down) override;
};

glm::vec2 vector2vec2(std::vector<float> const& v)
{
  assert(v.size() >= 2);
  return glm::vec2(v[0], v[1]);
}
glm::vec3 vector2vec3(std::vector<float> const& v)
{
  assert(v.size() >= 3);
  return glm::vec3(v[0], v[1], v[2]);
}
glm::vec4 vector2vec4(std::vector<float> const& v)
{
  assert(v.size() >= 4);
  return glm::vec4(v[0], v[1], v[2], v[3]);
}
glm::ivec2 vector2ivec2(std::vector<int32_t> const& v)
{
  assert(v.size() >= 2);
  return glm::ivec2(v[0], v[1]);
}
glm::ivec3 vector2ivec3(std::vector<int32_t> const& v)
{
  assert(v.size() >= 3);
  return glm::ivec3(v[0], v[1], v[2]);
}
glm::ivec4 vector2ivec4(std::vector<int32_t> const& v)
{
  assert(v.size() >= 4);
  return glm::ivec4(v[0], v[1], v[2], v[3]);
}
glm::uvec2 vector2uvec2(std::vector<uint32_t> const& v)
{
  assert(v.size() >= 2);
  return glm::uvec2(v[0], v[1]);
}
glm::uvec3 vector2uvec3(std::vector<uint32_t> const& v)
{
  assert(v.size() >= 3);
  return glm::uvec3(v[0], v[1], v[2]);
}
glm::uvec4 vector2uvec4(std::vector<uint32_t> const& v)
{
  assert(v.size() >= 4);
  return glm::uvec4(v[0], v[1], v[2], v[3]);
}
glm::vec2 vector2vec2(std::vector<double> const& v)
{
  assert(v.size() >= 2);
  return glm::vec2(v[0], v[1]);
}
glm::vec3 vector2vec3(std::vector<double> const& v)
{
  assert(v.size() >= 3);
  return glm::vec3(v[0], v[1], v[2]);
}
glm::vec4 vector2vec4(std::vector<double> const& v)
{
  assert(v.size() >= 4);
  return glm::vec4(v[0], v[1], v[2], v[3]);
}

void Shadows::parseArguments()
{
  assert(this != nullptr);
  auto arg = std::make_shared<argumentViewer::ArgumentViewer>(argc, argv);
  // modelName  =
  // arg->gets("--model","/media/windata/ft/prace/models/o/o.3ds","model file
  // name");
  modelName =
      arg->gets("--model", "/media/windata/ft/prace/models/2tri/2tri.3ds",
                "model file name");

  windowSize =
      vector2uvec2(arg->getu32v("--window-size", {512, 512}, "window size"));

  lightPosition = vector2vec4(
      arg->getf32v("--light", {0.f, 1000.f, 0.f, 1.f}, "light position"));

  cameraFovy = arg->getf32("--camera-fovy", 1.5707963267948966f,
                           "camera field of view in y direction");
  cameraNear = arg->getf32("--camera-near", 0.1f, "camera near plane position");
  cameraFar =
      arg->getf32("--camera-far", std::numeric_limits<float>::infinity(),
                  "camera far plane position");
  sensitivity =
      arg->getf32("--camera-sensitivity", 0.01f, "camera sensitivity");
  orbitZoomSpeed =
      arg->getf32("--camera-zoomSpeed", 0.2f, "orbit camera zoom speed");
  freeCameraSpeed = arg->getf32("--camera-speed", 1.f, "free camera speed");
  cameraType = arg->gets("--camera-type", "free", "orbit/free camera type");

  useShadows = !arg->isPresent("--no-shadows", "turns off shadows");
  verbose    = arg->isPresent("--verbose", "toggle verbose mode");
  methodName = arg->gets("--method", "",
                         "name of shadow method: "
                         "cubeShadowMapping/cssv/sintorn/rssv/vssv/cssvsoe");

  wavefrontSize =
      arg->getu32("--wavefrontSize", 0,
                  "warp/wavefront size, usually 32 for NVidia and 64 for AMD");

  maxMultiplicity =
      arg->getu32("--maxMultiplicity", 2,
                  "max number of triangles that share the same edge");
  svParams.zfail = arg->getu32("--zfail", 1, "shadow volumes zfail 0/1");

  cubeSMParams.resolution =
      arg->getu32("--shadowMap-resolution", 1024, "shadow map resolution");
  cubeSMParams.near =
      arg->getf32("--shadowMap-near", 0.1f, "shadow map near plane position");
  cubeSMParams.far =
      arg->getf32("--shadowMap-far", 1000.f, "shadow map far plane position");
  cubeSMParams.faces = arg->getu32("--shadowMap-faces", 6,
                                   "number of used cube shadow map faces");

  cssvParams.computeSidesWGS = arg->getu32(
      "--cssv-WGS", 64, "compute silhouette shadow volumes work group size");
  cssvParams.localAtomic =
      arg->getu32("--cssv-localAtomic", 1, "use local atomic instructions");
  cssvParams.cullSides =
      arg->getu32("--cssv-cullSides", 0,
                  "enables culling of sides that are outside of viewfrustum");
  cssvParams.usePlanes =
      arg->getu32("--cssv-usePlanes", 0,
                  "use triangle planes instead of opposite vertices");
  cssvParams.useInterleaving =
      arg->getu32("--cssv-useInterleaving", 0,
                  "reorder edge that so it is struct of arrays");
  std::cout << cssvParams.localAtomic << std::endl;

  cssvsoeParams.computeSidesWGS = arg->getu32(
      "--cssvsoe-WGS", 64, "compute silhouette shadow volumes work group size");

  vssvParams.usePlanes = arg->geti32("--vssv-usePlanes", 0,
                                     "use planes instead of opposite vertices");
  vssvParams.useStrips =
      arg->geti32("--vssv-useStrips", 1,
                  "use triangle strips for sides of shadow volumes 0/1");
  vssvParams.useAllOppositeVertices = arg->geti32(
      "--vssv-useAll", 0, "use all opposite vertices (even empty) 0/1");
  vssvParams.drawCapsSeparately =
      arg->geti32("--vssv-capsSeparate", 0, "draw caps using two draw calls");

  sintornParams.shadowFrustaPerWorkGroup =
      arg->geti32("--sintorn-frustumsPerWorkgroup", 1,
                  "nof triangles solved by work group");
  sintornParams.bias =
      arg->getf32("--sintorn-bias", 0.01f, "offset of triangle planes");
  sintornParams.discardBackFacing =
      arg->geti32("--sintorn-discardBackFacing", 1,
                  "discard light back facing fragments from hierarchical depth "
                  "texture construction");

  rssvParams.computeSilhouetteWGS =
      arg->geti32("--rssv-computeSilhouettesWGS", 64,
                  "workgroups size for silhouette computation");
  rssvParams.localAtomic =
      arg->geti32("--rssv-localAtomic", 1,
                  "use local atomic instructions in silhouette computation");
  rssvParams.cullSides               = arg->geti32("--rssv-cullSides", 0,
                                     "enables frustum culling of silhouettes");
  rssvParams.silhouettesPerWorkgroup = arg->geti32(
      "--rssv-silhouettesPerWorkgroup", 1,
      "number of silhouette edges that are compute by one workgroup");
  rssvParams.usePlanes =
      arg->geti32("--rssv-usePlanes", 0,
                  "use triangle planes instead of opposite vertices");

  testName           = arg->gets("--test", "", "name of test - fly or empty");
  testFlyKeyFileName = arg->gets(
      "--test-fly-keys", "",
      "filename containing fly keyframes - csv x,y,z,vx,vy,vz,ux,uy,uz");
  testFlyLength =
      arg->geti32("--test-fly-length", 1000, "number of measurements, 1000");
  testFramesPerMeasurement = arg->geti32(
      "--test-framesPerMeasurement", 5,
      "number of frames that is averaged per one measurement point");
  testOutputName =
      arg->gets("--test-output", "measurement", "name of output file");

  bool printHelp = arg->isPresent("-h", "prints this help");

  printHelp = printHelp || !arg->validate();
  if (printHelp) {
    std::cerr << arg->toStr();
    exit(0);
  }
}

void Shadows::initWavefrontSize()
{
  assert(this != nullptr);
  if (wavefrontSize == 0) {
    std::string renderer = std::string((char*)ge::gl::glGetString(GL_RENDERER));
    std::string vendor   = std::string((char*)ge::gl::glGetString(GL_VENDOR));
    std::cout << renderer << std::endl;
    std::cout << vendor << std::endl;
    if (vendor.find("AMD") != std::string::npos ||
        renderer.find("AMD") != std::string::npos)
      wavefrontSize = 64;
    else if (vendor.find("NVIDIA") != std::string::npos ||
             renderer.find("NVIDIA") != std::string::npos)
      wavefrontSize = 32;
    else {
      std::cerr << "WARNING: renderer is not NVIDIA or AMD, setting "
                   "wavefrontSize to 32"
                << std::endl;
      wavefrontSize = 32;
    }
  }
}

void Shadows::initCamera()
{
  assert(this != nullptr);
  if (cameraType == "orbit")
    cameraTransform = std::make_shared<basicCamera::OrbitCamera>();
  else if (cameraType == "free")
    cameraTransform = std::make_shared<basicCamera::FreeLookCamera>();
  else {
    std::cerr << "ERROR: --camera-type is incorrect" << std::endl;
    exit(0);
  }
  cameraProjection = std::make_shared<basicCamera::PerspectiveCamera>(
      cameraFovy, (float)windowSize.x / (float)windowSize.y, cameraNear,
      cameraFar);
}

void Shadows::init()
{
  parseArguments();

  window->setSize(windowSize.x,windowSize.y);

  ge::gl::glEnable(GL_DEPTH_TEST);
  ge::gl::glDepthFunc(GL_LEQUAL);
  ge::gl::glDisable(GL_CULL_FACE);
  ge::gl::glClearColor(0, 0, 0, 1);

  initWavefrontSize();

  if (testName == "fly" || testName == "grid") cameraType = "free";

  initCamera();

  gBuffer = std::make_shared<GBuffer>(windowSize.x, windowSize.y);

  model       = std::make_shared<Model>(modelName);
  renderModel = std::make_shared<RenderModel>(model);

  shadowMask = std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D, GL_R32F, 1,
                                                 windowSize.x, windowSize.y);
  shading    = std::make_shared<Shading>(gBuffer->color, gBuffer->position,
                                      gBuffer->normal, shadowMask);

  emptyVAO = std::make_shared<ge::gl::VertexArray>();

  if (methodName == "cubeShadowMapping")
    shadowMethod = std::make_shared<CubeShadowMapping>(
        shadowMask, windowSize, gBuffer->position, renderModel->nofVertices,
        renderModel->vertices, cubeSMParams);
  else if (methodName == "cssv")
    shadowMethod =
        std::make_shared<CSSV>(shadowMask, model, gBuffer->depth, svParams,
                               maxMultiplicity, cssvParams);
  else if (methodName == "cssvsoe")
    shadowMethod =
        std::make_shared<CSSVSOE>(shadowMask, model, gBuffer->depth, svParams,
                                  maxMultiplicity, cssvsoeParams);
  else if (methodName == "sintorn")
    shadowMethod = std::make_shared<Sintorn>(
        shadowMask, windowSize, gBuffer->depth, gBuffer->normal, model,
        wavefrontSize, sintornParams);
  else if (methodName == "rssv")
    shadowMethod =
        std::make_shared<RSSV>(shadowMask, windowSize, gBuffer->depth, model,
                               maxMultiplicity, rssvParams, wavefrontSize);
  else if (methodName == "vssv")
    shadowMethod =
        std::make_shared<VSSV>(shadowMask, model, gBuffer->depth, svParams,
                               maxMultiplicity, vssvParams);
  else
    useShadows = false;

  if (verbose)
    timeStamper = std::make_shared<TimeStamp>();
  else
    timeStamper = nullptr;  // std::make_shared<TimeStamp>(nullptr);
  if (shadowMethod) shadowMethod->timeStamp = timeStamper;

  if (testName == "fly" || testName == "grid") {
    if (shadowMethod != nullptr) {
      shadowMethod->timeStamp = timeStamper;
    }
  }

  drawPrimitive = std::make_shared<DrawPrimitive>(windowSize);
}

void Shadows::drawScene()
{
  if (timeStamper) timeStamper->begin();

  ge::gl::glViewport(0, 0, windowSize.x, windowSize.y);
  ge::gl::glEnable(GL_DEPTH_TEST);
  gBuffer->begin();
  ge::gl::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
                  GL_STENCIL_BUFFER_BIT);
  shadowMask->clear(0, GL_RED, GL_FLOAT);
  renderModel->draw(cameraProjection->getProjection() *
                    cameraTransform->getView());
  gBuffer->end();

  if (timeStamper) timeStamper->stamp("gBuffer");

  if (shadowMethod)
    shadowMethod->create(lightPosition, cameraTransform->getView(),
                         cameraProjection->getProjection());

  if (timeStamper) timeStamper->stamp("");
  ge::gl::glDisable(GL_DEPTH_TEST);
  shading->draw(lightPosition,
                glm::vec3(glm::inverse(cameraTransform->getView()) *
                          glm::vec4(0, 0, 0, 1)),
                useShadows);
  if (timeStamper) timeStamper->end("shading");
}

void Shadows::measure()
{
  assert(this != nullptr);
  if (testFlyKeyFileName == "") {
    std::cerr << "camera path file is empty" << std::endl;
    mainLoop->removeWindow(window->getId());
    return;
  }
  auto cameraPath = std::make_shared<CameraPath>(false, testFlyKeyFileName);
  std::map<std::string, float> measurement;
  timeStamper->setPrinter([&](std::vector<std::string> const& names,
                              std::vector<float> const&       values) {
    for (size_t i = 0; i < names.size(); ++i)
      if (names[i] != "") {
        if (measurement.count(names[i]) == 0) measurement[names[i]] = 0.f;
        measurement[names[i]] += values[i];
      }
  });

  std::vector<std::vector<std::string>> csv;
  for (size_t k = 0; k < testFlyLength; ++k) {
    auto keypoint = cameraPath->getKeypoint(float(k) / float(testFlyLength));
    auto flc =
        std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
    flc->setPosition(keypoint.position);
    flc->setRotation(keypoint.viewVector, keypoint.upVector);

    for (size_t f = 0; f < testFramesPerMeasurement; ++f) drawScene();

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
            x.second / float(testFramesPerMeasurement)));
    csv.push_back(line);
    measurement.clear();
    window->swap();
  }
  std::string output = testOutputName + ".csv";
  saveCSV(output, csv);
  mainLoop->removeWindow(window->getId());
}

void Shadows::draw()
{
  assert(this != nullptr);

  if (testName == "fly") {
    measure();
    return;
  }

  if (cameraType == "free") {
    auto freeLook =
        std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
    for (int a = 0; a < 3; ++a)
      freeLook->move(
          a, float(keyDown["d s"[a]] - keyDown["acw"[a]]) * freeCameraSpeed);
  }

  drawScene();

  // drawPrimitive->drawTexture(gBuffer->normal);
  //*
  if (methodName == "sintorn") {
    auto sintorn = std::dynamic_pointer_cast<Sintorn>(shadowMethod);
    if (keyDown['h']) drawPrimitive->drawTexture(sintorn->_HDT[0]);
    if (keyDown['j']) drawPrimitive->drawTexture(sintorn->_HDT[1]);
    if (keyDown['k']) drawPrimitive->drawTexture(sintorn->_HDT[2]);
    if (keyDown['l']) drawPrimitive->drawTexture(sintorn->_HDT[3]);

    if (keyDown['v']) sintorn->drawHST(0);
    if (keyDown['b']) sintorn->drawHST(1);
    if (keyDown['n']) sintorn->drawHST(2);
    if (keyDown['m']) sintorn->drawHST(3);
    if (keyDown[',']) sintorn->drawFinalStencilMask();
  }
  if (methodName == "rssv") {
    auto rssv = std::dynamic_pointer_cast<RSSV>(shadowMethod);
    if (keyDown['h']) drawPrimitive->drawTexture(rssv->_HDT[0]);
    if (keyDown['j']) drawPrimitive->drawTexture(rssv->_HDT[1]);
    if (keyDown['k']) drawPrimitive->drawTexture(rssv->_HDT[2]);
    if (keyDown['l']) drawPrimitive->drawTexture(rssv->_HDT[3]);
  }

  // */
  swap();
}

void testRSSVTiler(size_t w, size_t h, size_t warp)
{
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

std::string uvec2ToStr(glm::uvec2 const& v)
{
  std::stringstream ss;
  ss << v.x << "," << v.y;
  return ss.str();
}

void printTiling(size_t w, size_t h, size_t t)
{
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

int main(int argc, char* argv[])
{
  Shadows app{argc, argv};
  app.start();
  return EXIT_SUCCESS;
}

void Shadows::key(SDL_Event const& event, bool DOWN)
{
  keyDown[event.key.keysym.sym] = DOWN;
  if (DOWN && event.key.keysym.sym == 'p') {
    auto flc =
        std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
    if (!flc) return;
    auto rv   = flc->getRotation();
    auto pos  = flc->getPosition();
    auto up   = glm::normalize(glm::vec3(glm::row(rv, 1)));
    auto view = glm::normalize(-glm::vec3(glm::row(rv, 2)));
    std::cout << pos.x << "," << pos.y << "," << pos.z << ",";
    std::cout << view.x << "," << view.y << "," << view.z << ",";
    std::cout << up.x << "," << up.y << "," << up.z << std::endl;
  }
}

void Shadows::mouseMove(SDL_Event const& event)
{
  if (cameraType == "orbit") {
    if (event.motion.state & SDL_BUTTON_LMASK) {
      auto orbitCamera =
          std::dynamic_pointer_cast<basicCamera::OrbitCamera>(cameraTransform);
      if (orbitCamera) {
        orbitCamera->addXAngle(float(event.motion.yrel) * sensitivity);
        orbitCamera->addYAngle(float(event.motion.xrel) * sensitivity);
      }
    }
    if (event.motion.state & SDL_BUTTON_RMASK) {
      auto orbitCamera =
          std::dynamic_pointer_cast<basicCamera::OrbitCamera>(cameraTransform);
      if (orbitCamera)
        orbitCamera->addDistance(float(event.motion.yrel) * orbitZoomSpeed);
    }
    if (event.motion.state & SDL_BUTTON_MMASK) {
      auto orbitCamera =
          std::dynamic_pointer_cast<basicCamera::OrbitCamera>(cameraTransform);
      orbitCamera->addXPosition(+orbitCamera->getDistance() *
                                float(event.motion.xrel) / float(windowSize.x) *
                                2.f);
      orbitCamera->addYPosition(-orbitCamera->getDistance() *
                                float(event.motion.yrel) / float(windowSize.y) *
                                2.f);
    }
  }
  if (cameraType == "free") {
    auto freeCamera =
        std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(cameraTransform);
    if (event.motion.state & SDL_BUTTON_LMASK) {
      freeCamera->setAngle(
          1, freeCamera->getAngle(1) + float(event.motion.xrel) * sensitivity);
      freeCamera->setAngle(
          0, freeCamera->getAngle(0) + float(event.motion.yrel) * sensitivity);
    }
  }
}
