#include <RSSV/BuildStupidHierarchy.h>
#include <RSSV/StupidHierarchyShaders.h>
#include <RSSV/Hierarchy.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <util.h>
#include <Deferred.h>
#include <TimeStamp.h>
#include <geGL/StaticCalls.h>
#include <algorithm>
#include <divRoundUp.h>


using namespace std;
using namespace ge::gl;
using namespace glm;
using namespace rssv;



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
  auto ws = *vars.get<uvec2>("windowSize");
  auto warp = vars.getSizeT("wavefrontSize");
  printHierarchy(Hierarchy(uvec2(ws.x,ws.y),warp));
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
