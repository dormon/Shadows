#include <RSSV/BuildStupidHierarchy.h>
#include <RSSV/StupidHierarchyShaders.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <util.h>
#include <Deferred.h>
#include <TimeStamp.h>
#include <geGL/StaticCalls.h>
#include <algorithm>

using namespace std;
using namespace ge::gl;
using namespace glm;
using namespace rssv;

uvec2 divRoundUp(uvec2 const&a,uvec2 const&b){
  return uvec2(divRoundUp(a.x,b.x),divRoundUp(a.y,b.y));
}

uint32_t ilog2(uint32_t a){
  return static_cast<uint32_t>(glm::log2(static_cast<float>(a)));
}

class Hier{
  public:
    std::vector<uvec2>levelSize;
    std::vector<uvec2>fullTileSize;
    std::vector<uvec2>fullTileSizeInPixels;
    std::vector<vec2 >fullTileSizeInClipSpace;
    std::vector<uvec2>fullTileCount;
    std::vector<uvec2>borderTileSize;
    std::vector<uvec2>borderTileSizeInPixels;
    std::vector<vec2 >borderTileSizeInClipSpace;
    uint32_t nofLevels;
};

void printHier(Hier const&h){
  std::cerr << "level size: " << std::endl;
  for(auto const&x:h.levelSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile size: " << std::endl;
  for(auto const&x:h.fullTileSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile size in pixels: " << std::endl;
  for(auto const&x:h.fullTileSizeInPixels)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full size in clip space: " << std::endl;
  for(auto const&x:h.fullTileSizeInClipSpace)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "full tile count: " << std::endl;
  for(auto const&x:h.fullTileCount)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "border size: " << std::endl;
  for(auto const&x:h.borderTileSize)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "border size in pixels: " << std::endl;
  for(auto const&x:h.borderTileSizeInPixels)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;

  std::cerr << "border size in clip space: " << std::endl;
  for(auto const&x:h.borderTileSizeInClipSpace)
    std::cerr << x.x << " x " << x.y << std::endl;
  std::cerr << std::endl;


}

template<typename T>
void reverse(std::vector<T>&v){
  std::reverse(v.begin(),v.end());
}

Hier computeSizes(glm::uvec2 const&windowSize,uint32_t branchingFactor){
  glm::uvec2 fullTileSize;
  fullTileSize.y = 1<<(ilog2(branchingFactor)/2);
  fullTileSize.x = branchingFactor / fullTileSize.y;

  Hier result;
  uvec2 levelSize                 = windowSize;
  uvec2 fullTileSizeInPixels      = uvec2(1)  ;
  uvec2 fullTileCount                         ;
  uvec2 borderTileSize                        ;
  uvec2 borderTileSizeInPixels                ;
  vec2  fullTileSizeInClipSpace               ;
  vec2  borderTileSizeInClipSpace             ;

  auto const computeLevel = [&](){
    fullTileSizeInPixels      *= fullTileSize;
    fullTileCount              = windowSize / fullTileSizeInPixels;
    borderTileSize             = levelSize-fullTileCount*fullTileSize;
    borderTileSizeInPixels     = windowSize-fullTileCount*fullTileSizeInPixels;
    fullTileSizeInClipSpace    = vec2(fullTileSizeInPixels) / vec2(windowSize) * 2.f;
    borderTileSizeInClipSpace  = vec2(borderTileSizeInPixels) / vec2(windowSize) * 2.f;
    result.levelSize                .push_back(levelSize                );
    result.fullTileSize             .push_back(fullTileSize             );
    result.fullTileCount            .push_back(fullTileCount            );
    result.borderTileSize           .push_back(borderTileSize           );
    result.fullTileSizeInPixels     .push_back(fullTileSizeInPixels     );
    result.borderTileSizeInPixels   .push_back(borderTileSizeInPixels   );
    result.fullTileSizeInClipSpace  .push_back(fullTileSizeInClipSpace  );
    result.borderTileSizeInClipSpace.push_back(borderTileSizeInClipSpace);
  };

  while(levelSize.x > 1 || levelSize.y > 1){
    bool xDone = levelSize.x == 1;
    bool yDone = levelSize.y == 1;
    while(fullTileSize.x >= levelSize.x*2 && !yDone){
      fullTileSize.x/=2;
      fullTileSize.y*=2;
    }
    while(fullTileSize.y >= levelSize.y*2 && !xDone){
      fullTileSize.y/=2;
      fullTileSize.x*=2;
    }
    computeLevel();
    levelSize    = divRoundUp(levelSize,fullTileSize);
    fullTileSize = uvec2(fullTileSize.y,fullTileSize.x);
  }
  computeLevel();


  reverse(result.levelSize                );
  reverse(result.fullTileSize             );
  reverse(result.fullTileCount            );
  reverse(result.borderTileSize           );
  reverse(result.fullTileSizeInPixels     );
  reverse(result.fullTileSizeInClipSpace  );
  reverse(result.borderTileSizeInPixels   );
  reverse(result.borderTileSizeInClipSpace);
  return result;
}

void BuildStupidHierarchy::allocateHierarchy(){

  auto const windowSize    = vars.get<uvec2>("windowSize"    );
  auto const wavefrontSize = vars.getSizeT  ("wavefrontSize" );
  auto const alignment     = vars.getSizeT  ("rssv.alignment");

  auto nofPixels = windowSize->x * windowSize->y;
  for(;;){
    size_t const bufferSize       = align(nofPixels*sizeof(float),alignment)*6;
    size_t const alignedNofPixels = bufferSize / sizeof(float);
    nofPixelsPerLevel       .push_back(nofPixels);
    alignedNofPixelsPerLevel.push_back(alignedNofPixels);
    hierarchy               .push_back(make_shared<Buffer>(bufferSize));
    if(nofPixels == 1)break;
    nofPixels = divRoundUp(nofPixels,wavefrontSize);
  }
}

void BuildStupidHierarchy::createLevel0Program(){
  copyLevel0Program = make_shared<Program>(
      make_shared<Shader>(GL_COMPUTE_SHADER,"#version 450\n",
        Shader::define("WAVEFRONT_SIZE",static_cast<int>(vars.getSizeT("wavefrontSize"))),
        copyLevel0Src)
      );
}

void BuildStupidHierarchy::createNextLevelProgram(){
  buildNextLevelProgram = make_shared<Program>(
      make_shared<Shader>(GL_COMPUTE_SHADER,"#version 450\n",
        Shader::define("WAVEFRONT_SIZE",static_cast<int>(vars.getSizeT("wavefrontSize"))),
        buildNextLevelSrc)
      );
}

BuildStupidHierarchy::BuildStupidHierarchy(vars::Vars&vars):BuildHierarchy(vars){


  auto r = computeSizes(uvec2(512,512),32);
  printHier(r);
  exit(0);

  allocateHierarchy();
  createLevel0Program();
  createNextLevelProgram();
}


void BuildStupidHierarchy::copyLevel0(){
  auto&      depth         = vars.get<GBuffer>("gBuffer")->depth;
  auto const windowSize    = vars.get<uvec2>("windowSize");
  auto const nofPixels     = windowSize->x * windowSize->y;
  auto const wavefrontSize = vars.getSizeT("wavefrontSize");
  depth->bind(0);
  copyLevel0Program
    ->set2uiv   ("windowSize"      ,value_ptr(*windowSize)        )
    ->set1ui    ("nofPixels"       ,nofPixelsPerLevel.at(0)       )
    ->set1ui    ("alignedNofPixels",alignedNofPixelsPerLevel.at(0))
    ->bindBuffer("level"           ,hierarchy.at(0)               )
    ->dispatch  (getDispatchSize(nofPixels,wavefrontSize));
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void BuildStupidHierarchy::buildNextLevel(size_t i){
  auto const windowSize    = vars.get<uvec2>("windowSize");
  auto const wavefrontSize = vars.getSizeT("wavefrontSize");
  auto const nofPixels     = nofPixelsPerLevel.at(i);

  buildNextLevelProgram
    ->bindBuffer("inputLevel"            ,hierarchy.at(i  )               )
    ->bindBuffer("outputLevel"           ,hierarchy.at(i+1)               )
    ->set1ui    ("inputNofPixels"        ,nofPixelsPerLevel       .at(i  ))
    ->set1ui    ("inputAlignedNofPixels" ,alignedNofPixelsPerLevel.at(i  ))
    ->set1ui    ("outputAlignedNofPixels",alignedNofPixelsPerLevel.at(i+1))
    ->set1i     ("useBridges"            ,i==0                            )
    ->dispatch(getDispatchSize(nofPixels,wavefrontSize));
}

void BuildStupidHierarchy::build(){
  copyLevel0();
  if(vars.has("timeStamp"))vars.get<TimeStamp>("timeStamp")->stamp("copyLevel0");
  for(size_t i=0;i<hierarchy.size()-1;++i)
    buildNextLevel(i);
  glMemoryBarrier(GL_SHADER_STORAGE_BUFFER);
}
