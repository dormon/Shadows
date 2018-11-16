#include <Sintorn/HierarchicalDepth.h>
#include <Sintorn/HierarchyShaders.h>
#include <Barrier.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <Deferred.h>
#include <glm/gtc/type_ptr.hpp>
#include <Vars/Caller.h>

using namespace std;
using namespace ge::gl;

const size_t WRITEDEPTHTEXTURE_BINDING_DEPTH  = 0;
const size_t WRITEDEPTHTEXTURE_BINDING_HDT    = 1;
const size_t WRITEDEPTHTEXTURE_BINDING_NORMAL = 2;

const size_t HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT  = 0;
const size_t HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT = 1;


void writeDepth(vars::Vars&vars,glm::vec4 const&lightPosition){
  vars::Caller caller(vars,__FUNCTION__);
  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();
  auto const&tileCount        = vars.getVector<glm::uvec2>("sintorn.tileCount");

  auto program = vars.get<Program>("sintorn.writeDepthProgram");
  program->use();
  program->set2uiv("windowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));
  vars.get<GBuffer>("gBuffer")->depth->bind(WRITEDEPTHTEXTURE_BINDING_DEPTH);
  if(vars.getBool("sintorn.discardBackFacing")){
    vars.get<GBuffer>("gBuffer")->normal->bind(WRITEDEPTHTEXTURE_BINDING_NORMAL);
    program->set4fv("lightPosition",glm::value_ptr(lightPosition));
  }
  auto&HDT = vars.getVector<shared_ptr<Texture>>("sintorn.HDT");
  HDT[nofLevels-1]->bindImage(WRITEDEPTHTEXTURE_BINDING_HDT);
  glDispatchCompute(
      tileCount[nofLevels-2].x,
      tileCount[nofLevels-2].y,
      1);

  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void reduceDepthBuffer(vars::Vars&vars){
  vars::Caller caller(vars,__FUNCTION__);
  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();
  auto const&tileSizeInPixels = vars.getVector<glm::uvec2>("sintorn.tileSizeInPixels");
  auto const&usedTiles        = vars.getVector<glm::uvec2>("sintorn.usedTiles");

  auto program = vars.get<Program>("sintorn.hierarchicalDepthProgram");
  program->use();
  program->set2uiv("WindowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));
  program->set2uiv("TileDivisibility",glm::value_ptr(tileDivisibility.data()[0]),(GLsizei)nofLevels);
  program->set2uiv("TileSizeInPixels",glm::value_ptr(tileSizeInPixels.data()[0]),(GLsizei)nofLevels);

  auto&HDT = vars.getVector<shared_ptr<Texture>>("sintorn.HDT");
  for(int l=(int)nofLevels-2;l>=0;--l){
    program->set1ui("DstLevel",(unsigned)l);
    HDT[l+1]->bindImage(HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT );
    HDT[l  ]->bindImage(HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT);
    glDispatchCompute(usedTiles[l].x,usedTiles[l].y,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
  }
}

void allocateHierarchicalDepth(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.usedTiles"}))return;
  vars::Caller caller(vars,__FUNCTION__);

  auto const&usedTiles = vars.getVector<glm::uvec2>("sintorn.usedTiles");
  auto const nofLevels = usedTiles.size();

  auto&HDT = vars.reCreateVector<shared_ptr<Texture>>("sintorn.HDT");

  for(size_t l=0;l<nofLevels;++l){
    HDT.push_back(make_shared<Texture>(GL_TEXTURE_2D,GL_RG32F,1,usedTiles[l].x,usedTiles[l].y));
    HDT.back()->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    HDT.back()->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST);
    float data[2]={1,-1};
    HDT.back()->clear(0,GL_RG,GL_FLOAT,data);
  }
}

void createWriteDepthProgram(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.tileDivisibility","sintorn.discardBackFacing"}))return;
  vars::Caller caller(vars,__FUNCTION__);

  auto const&tileDivisibility    = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels           = tileDivisibility.size();
  //compile shader programs

  vars.reCreate<Program>("sintorn.writeDepthProgram",
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("LOCAL_TILE_SIZE_X"               ,int(tileDivisibility[nofLevels-1].x)),
        Shader::define("LOCAL_TILE_SIZE_Y"               ,int(tileDivisibility[nofLevels-1].y)),
        Shader::define("WRITEDEPTHTEXTURE_BINDING_DEPTH" ,int(WRITEDEPTHTEXTURE_BINDING_DEPTH              )),
        Shader::define("WRITEDEPTHTEXTURE_BINDING_HDT"   ,int(WRITEDEPTHTEXTURE_BINDING_HDT                )),
        Shader::define("WRITEDEPTHTEXTURE_BINDING_NORMAL",int(WRITEDEPTHTEXTURE_BINDING_NORMAL             )),
        Shader::define("DISCARD_BACK_FACING"             ,int(vars.getBool("sintorn.discardBackFacing")    )),
        sintorn::writeDepthSrc));
}


void createHierarchicalDepthProgram(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"wavefrontSize","sintorn.tileDivisibility"}))return;
  vars::Caller caller(vars,__FUNCTION__);

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();
  auto wavefrontSize = vars.getSizeT("wavefrontSize");

  vars.reCreate<Program>("sintorn.hierarchicalDepthProgram",
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("DO_NOT_COUNT_WITH_INFINITY"                                                                     ),
        Shader::define("WAVEFRONT_SIZE"                            ,uint32_t(wavefrontSize                             )),
        Shader::define("MAX_LEVELS"                                ,uint32_t(nofLevels                                 )),
        Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT" ,int     (HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT )),
        Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT",int     (HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT)),
        sintorn::hierarchicalDepthSrc));

}


void computeHierarchicalDepth(vars::Vars&vars,glm::vec4 const&lightPosition){
  vars::Caller caller(vars,__FUNCTION__);
  allocateHierarchicalDepth(vars);
  createWriteDepthProgram(vars);
  createHierarchicalDepthProgram(vars);

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();

  if(nofLevels<2)return;

  writeDepth(vars,lightPosition);
  reduceDepthBuffer(vars);
}
