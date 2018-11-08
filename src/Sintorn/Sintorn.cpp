#include<Sintorn/Sintorn.h>
#include<Sintorn/ComputeTileSizes.h>
#include<Sintorn/ShadowFrusta.h>
#include<Sintorn/HierarchicalDepth.h>
#include<geGL/StaticCalls.h>
#include<FastAdjacency.h>
#include<sstream>
#include<iomanip>
#include<util.h>
#include<Deferred.h>

size_t const UINT_BIT_SIZE            = 32;

const size_t DRAWHDB_BINDING_HDBIMAGE = 0;
const size_t DRAWHDB_BINDING_HDT      = 1;

size_t RASTERIZETEXTURE_BINDING_FINALSTENCILMASK = 0;
size_t RASTERIZETEXTURE_BINDING_HST              = 1;
size_t RASTERIZETEXTURE_BINDING_HDT              = 5;
size_t RASTERIZETEXTURE_BINDING_TRIANGLE_ID      = 9;
size_t RASTERIZETEXTURE_BINDING_SHADOWFRUSTA     = 0;

const size_t MERGETEXTURE_BINDING_HSTINPUT  = 0;
const size_t MERGETEXTURE_BINDING_HSTOUTPUT = 1;

const size_t WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK = 0;
const size_t WRITESTENCILTEXTURE_BINDING_HSTINPUT         = 1;

using namespace std;
using namespace ge::gl;

#include<Barrier.h>

#include<Sintorn/MergeShaders.h>

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

Sintorn::Sintorn(vars::Vars&vars):
  ShadowMethod(vars)
{
  assert(this!=nullptr);

  _shadowMask = vars.get<Texture>("shadowMask");

  _useUniformTileSizeInClipSpace=false;
  _useUniformTileDivisibility   =false;

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
  


  //compile shader programs
#include<Sintorn/Shaders.h>

  auto const wavefrontSize = vars.getSizeT("wavefrontSize");

  createWriteStencilProgram(vars);
  createMergeProgram(vars);

  ClearStencilProgram=make_shared<Program>(
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        clearStencilCompSrc));


  RASTERIZETEXTURE_BINDING_HDT         = RASTERIZETEXTURE_BINDING_HST+nofLevels;
  RASTERIZETEXTURE_BINDING_TRIANGLE_ID = RASTERIZETEXTURE_BINDING_HDT+nofLevels;

  string TileSizeInClipSpaceDefines="";
  if(_useUniformTileSizeInClipSpace)
    TileSizeInClipSpaceDefines+=Shader::define("USE_UNIFORM_TILE_SIZE_IN_CLIP_SPACE");
  else{
    for(unsigned l=0;l<nofLevels;++l){
      stringstream DefineName;
      DefineName<<"TILE_SIZE_IN_CLIP_SPACE"<<l;
      TileSizeInClipSpaceDefines+=Shader::define(DefineName.str(),2,glm::value_ptr(tileSizeInClipSpace[l]));
    }
  }
  string TileDivisibilityDefines="";
  if(_useUniformTileDivisibility)
    TileDivisibilityDefines+=Shader::define("USE_UNIFORM_TILE_DIVISIBILITY");
  else{
    for(unsigned l=0;l<nofLevels;++l){
      stringstream DefineName;
      DefineName<<"TILE_DIVISIBILITY"<<l;
      TileDivisibilityDefines+=Shader::define(DefineName.str(),2,glm::value_ptr(tileDivisibility[l]));
    }
  }
  RasterizeTextureProgram=make_shared<Program>(
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
        rasterizeTextureCompSrc));

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

class ComputePipeline{
  public:
    void operator()(){
      assert(this != nullptr);
      _program->use();
      for(auto const&x:_ssboBinding)
        get<BUFFER>(x)->bindRange(
            GL_SHADER_STORAGE_BUFFER,
            get<INDEX> (x)     ,
            get<OFFSET>(x)     ,
            get<SIZE>  (x)     );
      _program->dispatch(
          _nofGroups[0],
          _nofGroups[1],
          _nofGroups[2]);
    }
    ComputePipeline*setSSBO(
        string                    const&name  ,
        shared_ptr<Buffer>const&buffer){
      return setSSBO(name,buffer,0,buffer->getSize());
    }
    ComputePipeline*setSSBO(
        string                    const&name  ,
        shared_ptr<Buffer>const&buffer,
        GLintptr                       const&offset,
        GLsizei                        const&size  ){
      auto const&binding = _program->getBufferBinding(name);
      if(binding == Program::nonExistingBufferBinding)
        return this;
      while(binding > _ssboBinding.size())
        _ssboBinding.push_back(SSBOBinding(nullptr,0,0,0));
      _ssboBinding.push_back(SSBOBinding(buffer,binding,offset,size));
      return this;
    }
  protected:
    using SSBOBinding = tuple<shared_ptr<Buffer>,GLuint,GLintptr,GLsizei>;
    enum SSBOBindingParts{
      BUFFER = 0,
      INDEX  = 1,
      OFFSET = 2,
      SIZE   = 3,
    };
    shared_ptr<Program> _program      = nullptr;
    GLuint                           _nofGroups[3] = {1,1,1};
    vector<SSBOBinding>         _ssboBinding           ;
};

void Sintorn::RasterizeTexture(){
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

  RasterizeTextureProgram->use();

  if(_useUniformTileDivisibility)
    RasterizeTextureProgram->set2uiv("TileDivisibility",glm::value_ptr(tileDivisibility.data()[0]),(GLsizei)nofLevels);
  if(_useUniformTileSizeInClipSpace)
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

void mergeStencil(vars::Vars&vars){
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

void Sintorn::MergeTexture(){
  auto const&tileCount        = vars.getVector<glm::uvec2>("sintorn.tileCount");
  auto const nofLevels        = tileCount.size();

  mergeStencil(vars);

  //glFinish();
  //glMemoryBarrier(GL_ALL_BARRIER_BITS);

  auto WriteStencilTextureProgram = vars.get<Program>("sintorn.writeStencilProgram");
  WriteStencilTextureProgram->use();
  WriteStencilTextureProgram->set2uiv("WindowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));

  auto&HST = vars.getVector<std::shared_ptr<Texture>>("sintorn.HST");
  auto finalStencilMask = vars.get<Texture>("sintorn.finalStencilMask");

  finalStencilMask->bindImage(WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK);
  HST[nofLevels-1]->bindImage(WRITESTENCILTEXTURE_BINDING_HSTINPUT);

  //GLsync Sync=0;

  glDispatchCompute(
      tileCount[nofLevels-2].x,
      tileCount[nofLevels-2].y,
      1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
  //glFinish();
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
  RasterizeTexture();
  ifExistStamp("rasterize");
  MergeTexture();
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
