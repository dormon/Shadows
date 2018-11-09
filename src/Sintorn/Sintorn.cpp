#include<Sintorn/Sintorn.h>
#include<Sintorn/ComputeTileSizes.h>
#include<Sintorn/ShadowFrusta.h>
#include<Sintorn/HierarchicalDepth.h>
#include<Sintorn/MergeStencil.h>
#include<Sintorn/RasterizationShaders.h>
#include<geGL/StaticCalls.h>
#include<FastAdjacency.h>
#include<sstream>
#include<iomanip>
#include<util.h>
#include<Deferred.h>

const size_t DRAWHDB_BINDING_HDBIMAGE = 0;
const size_t DRAWHDB_BINDING_HDT      = 1;

size_t RASTERIZETEXTURE_BINDING_FINALSTENCILMASK = 0;
size_t RASTERIZETEXTURE_BINDING_HST              = 1;
size_t RASTERIZETEXTURE_BINDING_HDT              = 5;
size_t RASTERIZETEXTURE_BINDING_TRIANGLE_ID      = 9;
size_t RASTERIZETEXTURE_BINDING_SHADOWFRUSTA     = 0;

using namespace std;
using namespace ge::gl;

#include<Barrier.h>

void createRasterizationProgram(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"wavefrontSize","sintorn.tileDivisibility","sintorn.tileSizeInClipSpace","sintorn.useUniformTileDivisibility","sintorn.useUniformTileSizeInClipSpace"}))return;

  auto useUniformTileDivisibility    = vars.getBool("sintorn.useUniformTileDivisibility"   );
  auto useUniformTileSizeInClipSpace = vars.getBool("sintorn.useUniformTileSizeInClipSpace");

  auto const&tileDivisibility    = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels           = tileDivisibility.size();
  auto const&tileSizeInClipSpace = vars.getVector<glm::vec2>("sintorn.tileSizeInClipSpace");

  auto const wavefrontSize = vars.getSizeT("wavefrontSize");

  RASTERIZETEXTURE_BINDING_HDT         = RASTERIZETEXTURE_BINDING_HST+nofLevels;
  RASTERIZETEXTURE_BINDING_TRIANGLE_ID = RASTERIZETEXTURE_BINDING_HDT+nofLevels;

  string TileSizeInClipSpaceDefines="";
  if(useUniformTileSizeInClipSpace)
    TileSizeInClipSpaceDefines+=Shader::define("USE_UNIFORM_TILE_SIZE_IN_CLIP_SPACE");
  else{
    for(unsigned l=0;l<nofLevels;++l){
      stringstream DefineName;
      DefineName<<"TILE_SIZE_IN_CLIP_SPACE"<<l;
      TileSizeInClipSpaceDefines+=Shader::define(DefineName.str(),2,glm::value_ptr(tileSizeInClipSpace[l]));
    }
  }
  string TileDivisibilityDefines="";
  if(useUniformTileDivisibility)
    TileDivisibilityDefines+=Shader::define("USE_UNIFORM_TILE_DIVISIBILITY");
  else{
    for(unsigned l=0;l<nofLevels;++l){
      stringstream DefineName;
      DefineName<<"TILE_DIVISIBILITY"<<l;
      TileDivisibilityDefines+=Shader::define(DefineName.str(),2,glm::value_ptr(tileDivisibility[l]));
    }
  }
  vars.reCreate<Program>("sintorn.rasterizationProgram",
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("NUMBER_OF_LEVELS"            ,int(nofLevels                      )),
        Shader::define("NUMBER_OF_LEVELS_MINUS_ONE"  ,int(nofLevels-1                    )),
        Shader::define("WAVEFRONT_SIZE"              ,int(wavefrontSize                  )),
        Shader::define("SHADOWFRUSTUMS_PER_WORKGROUP",int(vars.getUint32("sintorn.shadowFrustaPerWorkGroup"))),
        TileSizeInClipSpaceDefines,
        TileDivisibilityDefines,
        Shader::define("RASTERIZETEXTURE_BINDING_FINALSTENCILMASK",int(RASTERIZETEXTURE_BINDING_FINALSTENCILMASK)),
        Shader::define("RASTERIZETEXTURE_BINDING_HST"             ,int(RASTERIZETEXTURE_BINDING_HST             )),
        Shader::define("RASTERIZETEXTURE_BINDING_HDT"             ,int(RASTERIZETEXTURE_BINDING_HDT             )),
        Shader::define("RASTERIZETEXTURE_BINDING_TRIANGLE_ID"     ,int(RASTERIZETEXTURE_BINDING_TRIANGLE_ID     )),
        Shader::define("RASTERIZETEXTURE_BINDING_SHADOWFRUSTA"    ,int(RASTERIZETEXTURE_BINDING_SHADOWFRUSTA    )),
        sintorn::rasterizationShader));
}

Sintorn::Sintorn(vars::Vars&vars):
  ShadowMethod(vars)
{
  assert(this!=nullptr);

  _shadowMask = vars.get<Texture>("shadowMask");

  vars.addBool("sintorn.useUniformTileDivisibility"   ,false);
  vars.addBool("sintorn.useUniformTileSizeInClipSpace",false);

  computeTileSizes(vars);

  auto const&tileDivisibility    = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels           = tileDivisibility.size();
  auto const&tileSizeInPixels    = vars.getVector<glm::uvec2>("sintorn.tileSizeInPixels");
  auto const&tileSizeInClipSpace = vars.getVector<glm::vec2>("sintorn.tileSizeInClipSpace");
  auto const&tileCount           = vars.getVector<glm::uvec2>("sintorn.tileCount");
  auto const&usedTiles           = vars.getVector<glm::uvec2>("sintorn.usedTiles");



  //*
  for(size_t l=0;l<nofLevels;++l)
    cerr<<"TileCount: "<<tileCount[l].x<<" "<<tileCount[l].y<<endl;
  for(size_t l=0;l<nofLevels;++l)
    cerr<<"UsedTiles: "<<usedTiles[l].x<<" "<<usedTiles[l].y<<endl;
  for(size_t l=0;l<nofLevels;++l)
    cerr<<"TileDivisibility: "<<tileDivisibility[l].x<<" "<<tileDivisibility[l].y<<endl;
  for(size_t l=0;l<nofLevels;++l)
    cerr<<"TileSizeInClip: "<<tileSizeInClipSpace[l].x<<" "<<tileSizeInClipSpace[l].y<<endl;
  for(unsigned l=0;l<nofLevels;++l)
    cerr<<"TileSizeInPixels: "<<tileSizeInPixels[l].x<<" "<<tileSizeInPixels[l].y<<endl;
  // */
  
  createRasterizationProgram(vars);

  //compile shader programs
#include<Sintorn/Shaders.h>


   _blitProgram = make_shared<Program>(
      make_shared<Shader>(GL_COMPUTE_SHADER  ,blitCompSrc));

  _drawHSTProgram = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,drawHSTVertSrc),
      make_shared<Shader>(GL_FRAGMENT_SHADER,drawHSTFragSrc));

  _drawFinalStencilMask = make_shared<Program>(
      make_shared<Shader>(GL_VERTEX_SHADER  ,drawHSTVertSrc),
      make_shared<Shader>(GL_FRAGMENT_SHADER,drawFinalStencilMaskFragSrc));


  _emptyVao=make_shared<VertexArray>();

  allocateHierarchicalStencil(vars);

}

Sintorn::~Sintorn(){
}

void rasterize(vars::Vars&vars){
  auto useUniformTileDivisibility    = vars.getBool("sintorn.useUniformTileDivisibility"   );
  auto useUniformTileSizeInClipSpace = vars.getBool("sintorn.useUniformTileSizeInClipSpace");

  auto const&tileDivisibility    = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const&tileSizeInClipSpace = vars.getVector<glm::vec2>("sintorn.tileSizeInClipSpace");
  auto const nofLevels = tileDivisibility.size();

  auto finalStencilMask = vars.get<Texture>("sintorn.finalStencilMask");
  finalStencilMask->clear(0,GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);

  auto&HST = vars.getVector<std::shared_ptr<Texture>>("sintorn.HST");
  //glClearTexImage(_finalStencilMask->getId(),0,GL_RED_INTEGER,GL_UNSIGNED_INT,NULL);
  for(size_t l=0;l<nofLevels;++l){
    HST[l]->clear(0,GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
    //glClearTexImage(_HST[l]->getId(),0,GL_RED_INTEGER,GL_UNSIGNED_INT,NULL);
  }
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

  auto RasterizeTextureProgram = vars.get<Program>("sintorn.rasterizationProgram");
  RasterizeTextureProgram->use();

  if(useUniformTileDivisibility)
    RasterizeTextureProgram->set2uiv("TileDivisibility",glm::value_ptr(tileDivisibility.data()[0]),(GLsizei)nofLevels);
  if(useUniformTileSizeInClipSpace)
    RasterizeTextureProgram->set2fv("TileSizeInClipSpace",glm::value_ptr(tileSizeInClipSpace.data()[0]),(GLsizei)nofLevels);

  RasterizeTextureProgram->set1ui("NumberOfTriangles",(uint32_t)vars.getSizeT("sintorn.nofTriangles"));

  vars.get<Buffer>("sintorn.shadowFrusta")->bindBase(GL_SHADER_STORAGE_BUFFER,0);

  auto&HDT = vars.getVector<shared_ptr<Texture>>("sintorn.HDT");
  for(size_t l=0;l<nofLevels;++l)
    HDT[l]->bind(GLuint(RASTERIZETEXTURE_BINDING_HDT+l));
  for(size_t l=0;l<nofLevels;++l)
    HST[l]->bindImage(GLuint(RASTERIZETEXTURE_BINDING_HST+l));

  finalStencilMask->bindImage(GLuint(RASTERIZETEXTURE_BINDING_FINALSTENCILMASK));

  vars.get<GBuffer>("gBuffer")->triangleIds->bind(static_cast<GLuint>(RASTERIZETEXTURE_BINDING_TRIANGLE_ID));

  
  size_t maxSize = 65536/2;
  size_t workgroups = getDispatchSize(vars.getSizeT("sintorn.nofTriangles"),vars.getUint32("sintorn.shadowFrustaPerWorkGroup"));
  size_t offset = 0;
  while(offset+maxSize<=workgroups){
    RasterizeTextureProgram->set1ui("triangleOffset",(uint32_t)offset);
    glDispatchCompute(GLuint(maxSize),1,1);
    offset += maxSize;
  }
  if(offset<workgroups){
    RasterizeTextureProgram->set1ui("triangleOffset",(uint32_t)offset);
    glDispatchCompute(GLuint(workgroups-offset),1,1);
  }

  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void Sintorn::create(
    glm::vec4 const&lightPosition,
    glm::mat4 const&view      ,
    glm::mat4 const&projection){
  assert(this!=nullptr);
  ifExistStamp("");
  computeHierarchicalDepth(vars,lightPosition);
  ifExistStamp("computeHDT");
  computeShadowFrusta(vars,lightPosition,projection*view);
  ifExistStamp("computeShadowFrusta");
  rasterize(vars);
  ifExistStamp("rasterize");
  mergeStencil(vars);
  ifExistStamp("merge");
  blit();
  ifExistStamp("blit");
}

void Sintorn::drawHST(size_t l){
  assert(this!=nullptr);
  _drawHSTProgram->use();
  auto&HST = vars.getVector<std::shared_ptr<Texture>>("sintorn.HST");
  HST[l]->bindImage(0);
  _emptyVao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  _emptyVao->unbind();
}

void Sintorn::drawFinalStencilMask(){
  assert(this!=nullptr);
  assert(_drawFinalStencilMask!=nullptr);
  assert(_drawFinalStencilMask!=nullptr);
  assert(_emptyVao!=nullptr);
  _drawFinalStencilMask->use();
  auto finalStencilMask = vars.get<Texture>("sintorn.finalStencilMask");
  finalStencilMask->bindImage(0);
  _emptyVao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  _emptyVao->unbind();
}

void Sintorn::blit(){
  assert(this!=nullptr);
  assert(_blitProgram!=nullptr);
  assert(_shadowMask!=nullptr);
  _blitProgram->use();
  auto finalStencilMask = vars.get<Texture>("sintorn.finalStencilMask");
  finalStencilMask->bindImage(0);
  _shadowMask->bindImage(1);
  _blitProgram->set2uiv("windowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));
  glDispatchCompute(
      (GLuint)getDispatchSize(vars.get<glm::uvec2>("windowSize")->x,8),
      (GLuint)getDispatchSize(vars.get<glm::uvec2>("windowSize")->y,8),1);
}
