#include<limits>
#include<string>
#include<limits>
#include<cmath>
#include<SDL2CPP/Window.h>
#include<geGL/geGL.h>
#include<geGL/OpenGLCommands.h>
#include<geGL/OpenGLContext.h>
#include<geGL/StaticCalls.h>
#include<geGL/VertexArray.h>

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtc/matrix_access.hpp>

#include<Deferred.h>
#include<Model.h>
#include<BasicCamera/OrbitCamera.h>
#include<BasicCamera/FreeLookCamera.h>
#include<ArgumentViewer/ArgumentViewer.h>
#include<BasicCamera/PerspectiveCamera.h>
#include<CameraPath.h>
#include<Shading.h>
#include<ShadowMethod.h>
#include<CubeShadowMapping.h>
#include<CSSV.h>
#include<TimeStamp.h>
#include<Sintorn.h>
#include<SintornTiles.h>
#include<DrawPrimitive.h>
#include<CSV.h>
#include<RSSV.h>
#include<VSSV.h>
#include<CSSVSOE.h>
#include<RSSVTiles.h>

struct Application{
  std::shared_ptr<sdl2cpp::MainLoop         >mainLoop         = nullptr;
  std::shared_ptr<sdl2cpp::Window           >window           = nullptr;
  std::shared_ptr<ge::gl::VertexArray       >emptyVAO         = nullptr;
  std::shared_ptr<GBuffer                   >gBuffer          = nullptr;
  std::shared_ptr<Model                     >model            = nullptr;
  std::shared_ptr<RenderModel               >renderModel      = nullptr;
  std::shared_ptr<basicCamera::CameraTransform >cameraTransform  = nullptr;
  std::shared_ptr<basicCamera::CameraProjection>cameraProjection = nullptr;
  std::shared_ptr<Shading                   >shading          = nullptr;
  std::shared_ptr<ge::gl::Texture           >shadowMask       = nullptr;
  std::shared_ptr<ShadowMethod              >shadowMethod     = nullptr;
  std::shared_ptr<DrawPrimitive             >drawPrimitive    = nullptr;
  std::shared_ptr<TimeStamp                 >timeStamper      = nullptr;
  glm::uvec2  windowSize          = glm::uvec2(512u,512u);

  std::string cameraType          = "orbit"           ;
  float       cameraFovy          = glm::radians(90.f);
  float       cameraNear          = 0.1f              ;
  float       cameraFar           = 1000.f            ;
  float       sensitivity         = 0.01f             ;
  float       orbitZoomSpeed      = 0.2f              ;
  float       freeCameraSpeed     = 1.f               ;

  glm::vec4   lightPosition       = glm::vec4(100.f,100.f,100.f,1.f);

  size_t      wavefrontSize       = 0;

  size_t      maxMultiplicity     = 2    ;

  ShadowVolumesParams     svParams     ;
  CubeShadowMappingParams cubeSMParams ;
  CSSVParams              cssvParams   ;
  CSSVSOEParams           cssvsoeParams;
  VSSVParams              vssvParams   ;
  SintornParams           sintornParams;
  RSSVParams              rssvParams   ;

  std::string testName                 = ""           ;
  std::string testFlyKeyFileName       = ""           ;
  size_t      testFlyLength            = 0            ;
  size_t      testFramesPerMeasurement = 5            ;
  std::string testOutputName           = "measurement";

  std::string modelName  = "";
  std::string methodName = "";
  bool verbose    = false;
  bool useShadows = true;

  bool init(int argc,char*argv[]);
  void parseArguments(int argc,char*argv[]);
  void initWavefrontSize();
  void initCamera();
  void draw();
  void measure();
  void drawScene();
  bool mouseMove(SDL_Event const&event);
  std::map<SDL_Keycode,bool>keyDown;
  template<bool DOWN>bool keyboard(SDL_Event const&event);
};

glm::vec2  vector2vec2 (std::vector<float   >const&v){assert(v.size()>=2);return glm::vec2 (v[0],v[1]          );}
glm::vec3  vector2vec3 (std::vector<float   >const&v){assert(v.size()>=3);return glm::vec3 (v[0],v[1],v[2]     );}
glm::vec4  vector2vec4 (std::vector<float   >const&v){assert(v.size()>=4);return glm::vec4 (v[0],v[1],v[2],v[3]);}
glm::ivec2 vector2ivec2(std::vector<int32_t >const&v){assert(v.size()>=2);return glm::ivec2(v[0],v[1]          );}
glm::ivec3 vector2ivec3(std::vector<int32_t >const&v){assert(v.size()>=3);return glm::ivec3(v[0],v[1],v[2]     );}
glm::ivec4 vector2ivec4(std::vector<int32_t >const&v){assert(v.size()>=4);return glm::ivec4(v[0],v[1],v[2],v[3]);}
glm::uvec2 vector2uvec2(std::vector<uint32_t>const&v){assert(v.size()>=2);return glm::uvec2(v[0],v[1]          );}
glm::uvec3 vector2uvec3(std::vector<uint32_t>const&v){assert(v.size()>=3);return glm::uvec3(v[0],v[1],v[2]     );}
glm::uvec4 vector2uvec4(std::vector<uint32_t>const&v){assert(v.size()>=4);return glm::uvec4(v[0],v[1],v[2],v[3]);}
glm::vec2  vector2vec2 (std::vector<double  >const&v){assert(v.size()>=2);return glm::vec2 (v[0],v[1]          );}
glm::vec3  vector2vec3 (std::vector<double  >const&v){assert(v.size()>=3);return glm::vec3 (v[0],v[1],v[2]     );}
glm::vec4  vector2vec4 (std::vector<double  >const&v){assert(v.size()>=4);return glm::vec4 (v[0],v[1],v[2],v[3]);}

void Application::parseArguments(int argc,char*argv[]){
  assert(this!=nullptr);
  auto arg = std::make_shared<argumentViewer::ArgumentViewer>(argc,argv);
  //this->modelName  = arg->gets("--model","/media/windata/ft/prace/models/o/o.3ds","model file name");
  this->modelName  = arg->gets("--model","/media/windata/ft/prace/models/2tri/2tri.3ds","model file name");

  this->windowSize = vector2uvec2(arg->getu32v("--window-size",{512,512},"window size" ));

  this->lightPosition       = vector2vec4(arg->getf32v("--light",{0.f,1000.f,0.f,1.f},"light position"));

  this->cameraFovy          = arg->getf32("--camera-fovy"       ,1.5707963267948966f                   ,"camera field of view in y direction");
  this->cameraNear          = arg->getf32("--camera-near"       ,0.1f                                  ,"camera near plane position"         );
  this->cameraFar           = arg->getf32("--camera-far"        ,std::numeric_limits<float>::infinity(),"camera far plane position"          );
  this->sensitivity         = arg->getf32("--camera-sensitivity",0.01f                                 ,"camera sensitivity"                 );
  this->orbitZoomSpeed      = arg->getf32("--camera-zoomSpeed"  ,0.2f                                  ,"orbit camera zoom speed"            );
  this->freeCameraSpeed     = arg->getf32("--camera-speed"      ,1.f                                   ,"free camera speed"                  );
  this->cameraType          = arg->gets  ("--camera-type"       ,"free"                                ,"orbit/free camera type"             );

  this->useShadows          = !arg->isPresent("--no-shadows",   "turns off shadows"                                                      );
  this->verbose             =  arg->isPresent("--verbose"   ,   "toggle verbose mode"                                                    );
  this->methodName          =  arg->gets     ("--method"    ,"","name of shadow method: cubeShadowMapping/cssv/sintorn/rssv/vssv/cssvsoe");

  this->wavefrontSize       = arg->getu32("--wavefrontSize",0,"warp/wavefront size, usually 32 for NVidia and 64 for AMD");

  this->maxMultiplicity     = arg->getu32("--maxMultiplicity" ,2 ,"max number of triangles that share the same edge"        );
  this->svParams.zfail      = arg->getu32("--zfail"           ,1 ,"shadow volumes zfail 0/1"                                );

  this->cubeSMParams.resolution = arg->getu32("--shadowMap-resolution",1024  ,"shadow map resolution"               );
  this->cubeSMParams.near       = arg->getf32("--shadowMap-near"      ,0.1f  ,"shadow map near plane position"      );
  this->cubeSMParams.far        = arg->getf32("--shadowMap-far"       ,1000.f,"shadow map far plane position"       );
  this->cubeSMParams.faces      = arg->getu32("--shadowMap-faces"     ,6     ,"number of used cube shadow map faces");

  this->cssvParams.computeSidesWGS = arg->getu32("--cssv-WGS"            ,64,"compute silhouette shadow volumes work group size"      );
  this->cssvParams.localAtomic     = arg->getu32("--cssv-localAtomic"    ,1 ,"use local atomic instructions"                           );
  this->cssvParams.cullSides       = arg->getu32("--cssv-cullSides"      ,0 ,"enables culling of sides that are outside of viewfrustum");
  this->cssvParams.usePlanes       = arg->getu32("--cssv-usePlanes"      ,0 ,"use triangle planes instead of opposite vertices"        );
  this->cssvParams.useInterleaving = arg->getu32("--cssv-useInterleaving",0 ,"reorder edge that so it is struct of arrays"             );

  this->cssvsoeParams.computeSidesWGS = arg->getu32("--cssvsoe-WGS",64,"compute silhouette shadow volumes work group size");

  this->vssvParams.usePlanes              = arg->geti32("--vssv-usePlanes"   ,0,"use planes instead of opposite vertices"            );
  this->vssvParams.useStrips              = arg->geti32("--vssv-useStrips"   ,1,"use triangle strips for sides of shadow volumes 0/1");
  this->vssvParams.useAllOppositeVertices = arg->geti32("--vssv-useAll"      ,0,"use all opposite vertices (even empty) 0/1"         );
  this->vssvParams.drawCapsSeparately     = arg->geti32("--vssv-capsSeparate",0,"draw caps using two draw calls"                     );

  this->sintornParams.shadowFrustaPerWorkGroup = arg->geti32("--sintorn-frustumsPerWorkgroup",1    ,"nof triangles solved by work group"                                              );
  this->sintornParams.bias                     = arg->getf32("--sintorn-bias"                ,0.01f,"offset of triangle planes"                                                       );
  this->sintornParams.discardBackFacing        = arg->geti32("--sintorn-discardBackFacing"   ,1    ,"discard light back facing fragments from hierarchical depth texture construction");

  this->rssvParams.computeSilhouetteWGS    = arg->geti32("--rssv-computeSilhouettesWGS"  ,64,"workgroups size for silhouette computation"                  );
  this->rssvParams.localAtomic             = arg->geti32("--rssv-localAtomic"            ,1 ,"use local atomic instructions in silhouette computation"     );
  this->rssvParams.cullSides               = arg->geti32("--rssv-cullSides"              ,0 ,"enables frustum culling of silhouettes"                      );
  this->rssvParams.silhouettesPerWorkgroup = arg->geti32("--rssv-silhouettesPerWorkgroup",1 ,"number of silhouette edges that are compute by one workgroup");
  this->rssvParams.usePlanes               = arg->geti32("--rssv-usePlanes"              ,0 ,"use triangle planes instead of opposite vertices"            );

  this->testName                 = arg->gets  ("--test"                     ,""           ,"name of test - fly or empty"                                    );
  this->testFlyKeyFileName       = arg->gets  ("--test-fly-keys"            ,""           ,"filename containing fly keyframes - csv x,y,z,vx,vy,vz,ux,uy,uz");
  this->testFlyLength            = arg->geti32("--test-fly-length"          ,1000         ,"number of measurements, 1000"                                   );
  this->testFramesPerMeasurement = arg->geti32("--test-framesPerMeasurement",5            ,"number of frames that is averaged per one measurement point"    );
  this->testOutputName           = arg->gets  ("--test-output"              ,"measurement","name of output file"                                            );

  bool printHelp = arg->isPresent("-h","prints this help");

  printHelp = printHelp || !arg->validate();
  if(printHelp){
    std::cerr<<arg->toStr();
    exit(0);
  }
}

void Application::initWavefrontSize(){
  assert(this!=nullptr);
  if(this->wavefrontSize==0){
    std::string renderer = std::string((char*)ge::gl::glGetString(GL_RENDERER));
    std::string vendor   = std::string((char*)ge::gl::glGetString(GL_VENDOR  ));
    std::cout << renderer << std::endl;
    std::cout << vendor   << std::endl;
    if     (vendor.find("AMD"   ) != std::string::npos || renderer.find("AMD"   ) != std::string::npos)
      this->wavefrontSize = 64;
    else if(vendor.find("NVIDIA") != std::string::npos || renderer.find("NVIDIA") != std::string::npos)
      this->wavefrontSize = 32;
    else{
      std::cerr<<"WARNING: renderer is not NVIDIA or AMD, setting wavefrontSize to 32"<<std::endl;
      this->wavefrontSize = 32;
    }
  }
}

void Application::initCamera(){
  assert(this!=nullptr);
  if     (this->cameraType == "orbit")
    this->cameraTransform = std::make_shared<basicCamera::OrbitCamera>();
  else if(this->cameraType == "free")
    this->cameraTransform = std::make_shared<basicCamera::FreeLookCamera>();
  else{
    std::cerr<<"ERROR: --camera-type is incorrect"<<std::endl;
    exit(0);
  }
  this->cameraProjection = std::make_shared<basicCamera::PerspectiveCamera>(this->cameraFovy,(float)this->windowSize.x/(float)this->windowSize.y,this->cameraNear,this->cameraFar);
}

bool Application::init(int argc,char*argv[]){
  this->parseArguments(argc,argv);

  this->mainLoop = std::make_shared<sdl2cpp::MainLoop>();
  this->mainLoop->setIdleCallback(std::bind(&Application::draw,this));
  this->window   = std::make_shared<sdl2cpp::Window>(this->windowSize.x,this->windowSize.y);
  this->window->setEventCallback(SDL_MOUSEMOTION,std::bind(&Application::mouseMove      ,this,std::placeholders::_1));
  this->window->setEventCallback(SDL_KEYDOWN    ,std::bind(&Application::keyboard<true> ,this,std::placeholders::_1));
  this->window->setEventCallback(SDL_KEYUP      ,std::bind(&Application::keyboard<false>,this,std::placeholders::_1));
  this->window->createContext("rendering",450u,sdl2cpp::Window::CORE,sdl2cpp::Window::DEBUG);
  this->mainLoop->addWindow("primaryWindow",this->window);
  this->window->makeCurrent("rendering");
  ge::gl::init(SDL_GL_GetProcAddress);
  ge::gl::setHighDebugMessage();

  ge::gl::glEnable(GL_DEPTH_TEST);
  ge::gl::glDepthFunc(GL_LEQUAL);
  ge::gl::glDisable(GL_CULL_FACE);
  ge::gl::glClearColor(0,0,0,1);

  this->initWavefrontSize();

  if(this->testName == "fly" || this->testName == "grid")
    this->cameraType = "free";

  this->initCamera();

  this->gBuffer = std::make_shared<GBuffer>(this->windowSize.x,this->windowSize.y);

  this->model = std::make_shared<Model>(this->modelName);
  this->renderModel = std::make_shared<RenderModel>(this->model);

  this->shadowMask = std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_R32F,1,this->windowSize.x,this->windowSize.y);
  this->shading = std::make_shared<Shading>(this->gBuffer->color,this->gBuffer->position,this->gBuffer->normal,this->shadowMask);

  this->emptyVAO = std::make_shared<ge::gl::VertexArray>();

  if     (this->methodName=="cubeShadowMapping")
    this->shadowMethod = std::make_shared<CubeShadowMapping>(
        this->shadowMask              ,
        this->windowSize              ,
        this->gBuffer->position       ,
        this->renderModel->nofVertices,
        this->renderModel->vertices   ,
        this->cubeSMParams            );
  else if(this->methodName=="cssv")
    this->shadowMethod = std::make_shared<CSSV>(
        this->shadowMask     ,
        this->model          ,
        this->gBuffer->depth ,
        this->svParams       ,
        this->maxMultiplicity,
        this->cssvParams     );
  else if(this->methodName=="cssvsoe")
    this->shadowMethod = std::make_shared<CSSVSOE>(
        this->shadowMask     ,
        this->model          ,
        this->gBuffer->depth ,
        this->svParams       ,
        this->maxMultiplicity,
        this->cssvsoeParams  );
  else if(this->methodName=="sintorn")
    this->shadowMethod = std::make_shared<Sintorn>(
        this->shadowMask     ,
        this->windowSize     ,
        this->gBuffer->depth ,
        this->gBuffer->normal,
        this->model          ,
        this->wavefrontSize  ,
        this->sintornParams  );
  else if(this->methodName=="rssv")
    this->shadowMethod = std::make_shared<RSSV>(
        this->shadowMask     ,
        this->windowSize     ,
        this->gBuffer->depth ,
        this->model          ,
        this->maxMultiplicity,
        this->rssvParams     ,
        this->wavefrontSize  );
  else if(this->methodName=="vssv")
    this->shadowMethod = std::make_shared<VSSV>(
        this->shadowMask     ,
        this->model          ,
        this->gBuffer->depth ,
        this->svParams       ,
        this->maxMultiplicity,
        this->vssvParams     );
  else
    this->useShadows = false;

  if(this->verbose)
    this->timeStamper = std::make_shared<TimeStamp>();
  else
    this->timeStamper = std::make_shared<TimeStamp>(nullptr);
  if(this->shadowMethod)
    this->shadowMethod->timeStamp = this->timeStamper;

  if(this->testName == "fly" || this->testName == "grid"){
    if(this->shadowMethod!=nullptr){
      this->shadowMethod->timeStamp = this->timeStamper;
    }
  }

  this->drawPrimitive = std::make_shared<DrawPrimitive>(this->windowSize);

  return true;
}

void Application::drawScene(){
  if(this->timeStamper)this->timeStamper->begin();

  ge::gl::glViewport(0,0,this->windowSize.x,this->windowSize.y);
  ge::gl::glEnable(GL_DEPTH_TEST);
  this->gBuffer->begin();
  ge::gl::glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
  this->shadowMask->clear(0,GL_RED,GL_FLOAT);
  this->renderModel->draw(this->cameraProjection->getProjection()*this->cameraTransform->getView());
  this->gBuffer->end();

  if(this->timeStamper)this->timeStamper->stamp("gBuffer");

  if(this->shadowMethod)
    this->shadowMethod->create(this->lightPosition,this->cameraTransform->getView(),this->cameraProjection->getProjection());

  if(this->timeStamper)this->timeStamper->stamp("");
  ge::gl::glDisable(GL_DEPTH_TEST);
  this->shading->draw(this->lightPosition,glm::vec3(glm::inverse(this->cameraTransform->getView())*glm::vec4(0,0,0,1)),this->useShadows);
  if(this->timeStamper)this->timeStamper->end("shading");
}

void Application::measure(){
  assert(this!=nullptr);
  if(this->testFlyKeyFileName==""){
    ge::core::printError(GE_CORE_FCENAME,"camera path file is empty");
    this->mainLoop->removeWindow(this->window->getId());
    return;
  }
  auto cameraPath = std::make_shared<CameraPath>(false,this->testFlyKeyFileName);
  std::map<std::string,float>measurement;
  this->timeStamper->setPrinter([&](std::vector<std::string>const&names,std::vector<float>const&values){
      for(size_t i=0;i<names.size();++i)
      if(names[i]!=""){
      if(measurement.count(names[i])==0)measurement[names[i]]=0.f;
      measurement[names[i]]+=values[i];
      }});

  std::vector<std::vector<std::string>>csv;
  for(size_t k=0;k<this->testFlyLength;++k){
    auto keypoint = cameraPath->getKeypoint(float(k)/float(this->testFlyLength));
    auto flc = std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(this->cameraTransform);
    flc->setPosition(keypoint.position);
    flc->setRotation(keypoint.viewVector,keypoint.upVector);

    for(size_t f=0;f<this->testFramesPerMeasurement;++f)
      this->drawScene();

    std::vector<std::string>line;
    if(csv.size()==0){
      line.push_back("frame");
      for(auto const&x:measurement)
        if(x.first!="")line.push_back(x.first);
      csv.push_back(line);
      line.clear();
    }
    line.push_back(ge::core::value2str(k));
    for(auto const&x:measurement)
      if(x.first!="")
        line.push_back(ge::core::value2str(x.second/float(this->testFramesPerMeasurement)));
    csv.push_back(line);
    measurement.clear();
    this->window->swap();
  }
  std::string output = this->testOutputName+".csv";
  saveCSV(output,csv);
  this->mainLoop->removeWindow(this->window->getId());
}

void Application::draw(){
  assert(this!=nullptr);

  if(this->testName == "fly"){
    this->measure();
    return;
  }

  if(this->cameraType == "free"){
    auto freeLook = std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(this->cameraTransform);
    for(int a=0;a<3;++a)freeLook->move(a,float(this->keyDown["d s"[a]]-this->keyDown["acw"[a]])*this->freeCameraSpeed);
  }

  this->drawScene();

  //this->drawPrimitive->drawTexture(this->gBuffer->normal);
  //*
  if(this->methodName == "sintorn"){
    auto sintorn = std::dynamic_pointer_cast<Sintorn>(this->shadowMethod);
    if(this->keyDown['h'])this->drawPrimitive->drawTexture(sintorn->_HDT[0]);
    if(this->keyDown['j'])this->drawPrimitive->drawTexture(sintorn->_HDT[1]);
    if(this->keyDown['k'])this->drawPrimitive->drawTexture(sintorn->_HDT[2]);
    if(this->keyDown['l'])this->drawPrimitive->drawTexture(sintorn->_HDT[3]);

    if(this->keyDown['v'])sintorn->drawHST(0);
    if(this->keyDown['b'])sintorn->drawHST(1);
    if(this->keyDown['n'])sintorn->drawHST(2);
    if(this->keyDown['m'])sintorn->drawHST(3);
    if(this->keyDown[','])sintorn->drawFinalStencilMask();
  }
  if(this->methodName == "rssv"){
    auto rssv = std::dynamic_pointer_cast<RSSV>(this->shadowMethod);
    if(this->keyDown['h'])this->drawPrimitive->drawTexture(rssv->_HDT[0]);
    if(this->keyDown['j'])this->drawPrimitive->drawTexture(rssv->_HDT[1]);
    if(this->keyDown['k'])this->drawPrimitive->drawTexture(rssv->_HDT[2]);
    if(this->keyDown['l'])this->drawPrimitive->drawTexture(rssv->_HDT[3]);
  }

  // */
  this->window->swap();
}

void testRSSVTiler(size_t w,size_t h,size_t warp){
  auto win  = glm::uvec2(w,h);
  std::cout << "window: " << win.x << "x" << win.y << "warp: " << warp << std::endl;
  auto d = rssvGetMaxUpperTileDivisibility(win,warp);
  for(auto const&x:d)
    std::cout << x.x << "x" << x.y <<std::endl;
  auto dd = d;
  for(size_t i=1;i<dd.size();++i)
    dd[dd.size()-1-i]*=dd[dd.size()-i];
  for(auto const&x:dd)
    std::cout << "pix: " << x.x << "x" << x.y <<std::endl;
  std::cout << std::endl;
}

std::string uvec2ToStr(glm::uvec2 const&v){
  std::stringstream ss;
  ss << v.x << "," << v.y;
  return ss.str();
}

void printTiling(size_t w,size_t h,size_t t){
  RSSVTilingSizes tiling(glm::uvec2(w,h),t);
  std::cout << w << " x " << h << " : " << t << std::endl;
  for(size_t i=0;i<tiling.borderTileDivisibilityIntoTiles.size();++i){
    std::cout << "FULL___TILE_DIVISIBILITY_INTO_PIXELS    (" <<  uvec2ToStr(tiling.full__TileDivisibilityIntoPixels   .at(i)) << ")" << std::endl;
    std::cout << "FULL___TILE_DIVISIBILITY_INTO_TILES     (" <<  uvec2ToStr(tiling.full__TileDivisibilityIntoTiles    .at(i)) << ")" << std::endl;
    std::cout << "BORDER_TILE_DIVISIBILITY_INTO_PIXELS    (" <<  uvec2ToStr(tiling.borderTileDivisibilityIntoPixels   .at(i)) << ")" << std::endl;
    std::cout << "BORDER_TILE_DIVISIBILITY_INTO_TILES     (" <<  uvec2ToStr(tiling.borderTileDivisibilityIntoTiles    .at(i)) << ")" << std::endl;
    std::cout << "BORDER_TILE_DIVISIBILITY_INTO_FULL_TILES(" <<  uvec2ToStr(tiling.borderTileDivisibilityIntoFullTiles.at(i)) << ")" << std::endl;
    std::cout << "HDT_SIZE                                (" <<  uvec2ToStr(tiling.hdtSize                            .at(i)) << ")" << std::endl;
    std::cout << std::endl;
  }
}

int main(int argc,char*argv[]){
#if 0
  printTiling(1024,1024,32);
  return 0;
#endif
  /*
  testRSSVTiler(1024,768,32);
  testRSSVTiler(64,2,32);
  testRSSVTiler(512,512,64);
  testRSSVTiler(1366,768,64);
  testRSSVTiler(1920,768,64);
  testRSSVTiler(1366,768,32);
  testRSSVTiler(1920,1080,32);
  testRSSVTiler(1920,1,32);
  testRSSVTiler(512,512,64);
  return 0;
  // */
  Application app;
  if(!app.init(argc,argv))return EXIT_FAILURE;
  (*app.mainLoop)();
  return EXIT_SUCCESS;
}


template<bool DOWN>bool Application::keyboard(SDL_Event const&event){
  this->keyDown[event.key.keysym.sym] = DOWN;
  if(DOWN && event.key.keysym.sym=='p'){
    auto flc = std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(this->cameraTransform);
    if(!flc)return true;
    auto rv = flc->getRotation();
    auto pos = flc->getPosition();
    auto up = glm::normalize(glm::vec3(glm::row(rv,1)));
    auto view = glm::normalize(-glm::vec3(glm::row(rv,2)));
    std::cout<< pos .x<<","<<pos .y<<","<<pos .z<<",";
    std::cout<< view.x<<","<<view.y<<","<<view.z<<",";
    std::cout<< up  .x<<","<<up  .y<<","<<up  .z<<std::endl;
  }
  return true;
}

bool Application::mouseMove(SDL_Event const&event){
  if(this->cameraType == "orbit"){
    if(event.motion.state & SDL_BUTTON_LMASK){
      auto orbitCamera = std::dynamic_pointer_cast<basicCamera::OrbitCamera>(this->cameraTransform);
      if(orbitCamera){
        orbitCamera->addXAngle(float(event.motion.yrel)*this->sensitivity);
        orbitCamera->addYAngle(float(event.motion.xrel)*this->sensitivity);
      }
    }
    if(event.motion.state & SDL_BUTTON_RMASK){
      auto orbitCamera = std::dynamic_pointer_cast<basicCamera::OrbitCamera>(this->cameraTransform);
      if(orbitCamera)
        orbitCamera->addDistance(float(event.motion.yrel)*this->orbitZoomSpeed);
    }
    if(event.motion.state & SDL_BUTTON_MMASK){
      auto orbitCamera = std::dynamic_pointer_cast<basicCamera::OrbitCamera>(this->cameraTransform);
      orbitCamera->addXPosition(+orbitCamera->getDistance()*float(event.motion.xrel)/float(this->windowSize.x)*2.f);
      orbitCamera->addYPosition(-orbitCamera->getDistance()*float(event.motion.yrel)/float(this->windowSize.y)*2.f);
    }
  }
  if(this->cameraType == "free"){
    auto freeCamera = std::dynamic_pointer_cast<basicCamera::FreeLookCamera>(this->cameraTransform);
    if(event.motion.state & SDL_BUTTON_LMASK){
      freeCamera->setAngle(1,freeCamera->getAngle(1)+float(event.motion.xrel)*this->sensitivity);
      freeCamera->setAngle(0,freeCamera->getAngle(0)+float(event.motion.yrel)*this->sensitivity);
    }
  }
  return true;
}
