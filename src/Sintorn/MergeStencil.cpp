#include <Sintorn/MergeStencil.h>
#include <Sintorn/MergeShaders.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Barrier.h>

using namespace ge::gl;
using namespace std;

size_t const UINT_BIT_SIZE            = 32;

const size_t MERGETEXTURE_BINDING_HSTINPUT  = 0;
const size_t MERGETEXTURE_BINDING_HSTOUTPUT = 1;

const size_t WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK = 0;
const size_t WRITESTENCILTEXTURE_BINDING_HSTINPUT         = 1;


void createMergeProgram(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"wavefrontSize"}))return;
  
  auto const wavefrontSize = vars.getSizeT("wavefrontSize");
  vars.add<Program>("sintorn.mergeProgram",
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("WAVEFRONT_SIZE"                ,uint32_t(wavefrontSize          )),
        Shader::define("MERGETEXTURE_BINDING_HSTINPUT" ,int     (MERGETEXTURE_BINDING_HSTINPUT )),
        Shader::define("MERGETEXTURE_BINDING_HSTOUTPUT",int     (MERGETEXTURE_BINDING_HSTOUTPUT)),
        sintorn::mergeShader));
}

void createWriteStencilProgram(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.tileDivisibility"}))return;

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels = tileDivisibility.size();

  vars.reCreate<Program>("sintorn.writeStencilProgram",
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("LOCAL_TILE_SIZE_X"                           ,int(tileDivisibility[nofLevels-1].x)),
        Shader::define("LOCAL_TILE_SIZE_Y"                           ,int(tileDivisibility[nofLevels-1].y)),
        Shader::define("WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK",int(WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK )),
        Shader::define("WRITESTENCILTEXTURE_BINDING_HSTINPUT"        ,int(WRITESTENCILTEXTURE_BINDING_HSTINPUT         )),
        sintorn::writeStencilShader));
}

void allocateHierarchicalStencil(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"windowSize","wavefrontSize","sintorn.usedTiles"}))return;

  auto const windowSize    = *vars.get<glm::uvec2>("windowSize");
  auto const wavefrontSize = vars.getSizeT("wavefrontSize");
  auto const&usedTiles     = vars.getVector<glm::uvec2>("sintorn.usedTiles");
  auto const nofLevels     = usedTiles.size();


  auto finalStencilMask = vars.reCreate<Texture>("sintorn.finalStencilMask",GL_TEXTURE_2D,GL_R32UI,1,windowSize.x,windowSize.y);
  finalStencilMask->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
  finalStencilMask->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST);

  size_t RESULT_LENGTH_IN_UINT=wavefrontSize/UINT_BIT_SIZE;
  if(RESULT_LENGTH_IN_UINT==0)RESULT_LENGTH_IN_UINT=1;

  auto&HST = vars.reCreateVector<std::shared_ptr<Texture>>("sintorn.HST");
  for(size_t l=0;l<nofLevels;++l){
    HST.push_back(make_shared<Texture>(GL_TEXTURE_2D,GL_R32UI,1,GLsizei(usedTiles[l].x*RESULT_LENGTH_IN_UINT),usedTiles[l].y));
    HST.back()->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    HST.back()->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST);
    uint8_t data[2] = {0,0};
    HST.back()->clear(0,GL_RG_INTEGER,GL_UNSIGNED_BYTE,data);
    //glClearTexImage(_HDT.back()->getId(),0,GL_RG,GL_UNSIGNED_INT,&data);
  }
}

void propagateStencil(vars::Vars&vars){
  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const&tileSizeInPixels = vars.getVector<glm::uvec2>("sintorn.tileSizeInPixels");
  auto const&tileCount        = vars.getVector<glm::uvec2>("sintorn.tileCount");
  auto const nofLevels        = tileDivisibility.size();

  auto program = vars.get<Program>("sintorn.mergeProgram");
  program->use();
  program->set2uiv("WindowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));

  auto&HST = vars.getVector<std::shared_ptr<Texture>>("sintorn.HST");
  GLsync Sync=0;
  for(size_t l=0;l<nofLevels-1;++l){
    program->set2uiv("DstTileSizeInPixels",glm::value_ptr(tileSizeInPixels[l]));
    program->set2uiv("DstTileDivisibility",glm::value_ptr(tileDivisibility[l]));

    HST[l  ]->bindImage(MERGETEXTURE_BINDING_HSTINPUT);
    HST[l+1]->bindImage(MERGETEXTURE_BINDING_HSTOUTPUT);
    if(l>0){
      glClientWaitSync(Sync,0,GL_TIMEOUT_IGNORED);
      glDeleteSync(Sync);
    }
    glDispatchCompute(l==0?1:tileCount[l-1].x,l==0?1:tileCount[l-1].y,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    if(l<nofLevels
        //*
        -1
        // */
      )Sync=glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
  }
  glClientWaitSync(Sync,0,GL_TIMEOUT_IGNORED);
  glDeleteSync(Sync);
}

void writeStencil(vars::Vars&vars){
  auto const&tileCount        = vars.getVector<glm::uvec2>("sintorn.tileCount");
  auto const nofLevels        = tileCount.size();

  auto WriteStencilTextureProgram = vars.get<Program>("sintorn.writeStencilProgram");
  WriteStencilTextureProgram->use();
  WriteStencilTextureProgram->set2uiv("WindowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));

  auto&HST = vars.getVector<std::shared_ptr<Texture>>("sintorn.HST");
  auto finalStencilMask = vars.get<Texture>("sintorn.finalStencilMask");

  finalStencilMask->bindImage(WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK);
  HST[nofLevels-1]->bindImage(WRITESTENCILTEXTURE_BINDING_HSTINPUT);

  glDispatchCompute(
      tileCount[nofLevels-2].x,
      tileCount[nofLevels-2].y,
      1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void mergeStencil(vars::Vars&vars){
  createWriteStencilProgram(vars);
  createMergeProgram(vars);
  allocateHierarchicalStencil(vars);
  propagateStencil(vars);
  writeStencil(vars);
}
