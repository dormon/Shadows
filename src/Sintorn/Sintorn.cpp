#include<Sintorn/Sintorn.h>
#include<Sintorn/Tiles.h>
#include<Sintorn/ShadowFrusta.h>
#include<Sintorn/HierarchyShaders.h>
#include<geGL/StaticCalls.h>
#include<FastAdjacency.h>
#include<sstream>
#include<iomanip>
#include<util.h>
#include<Deferred.h>

size_t const UINT_BIT_SIZE            = 32;

const size_t DRAWHDB_BINDING_HDBIMAGE = 0;
const size_t DRAWHDB_BINDING_HDT      = 1;

const size_t WRITEDEPTHTEXTURE_BINDING_DEPTH  = 0;
const size_t WRITEDEPTHTEXTURE_BINDING_HDT    = 1;
const size_t WRITEDEPTHTEXTURE_BINDING_NORMAL = 2;

const size_t HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT  = 0;
const size_t HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT = 1;

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

void computeTileDivisibility(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"wavefrontSize","windowSize"}))return;

  auto&tileDivisibility = vars.reCreateVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const windowSize = *vars.get<glm::uvec2>("windowSize");
  auto const wavefrontSize = vars.getSizeT("wavefrontSize");

  chooseTileSizes(tileDivisibility,windowSize,wavefrontSize);
}

void computeTileSizeInPixel(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.tileDivisibility"}))return;

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels = tileDivisibility.size();

  auto&tileSizeInPixels = vars.reCreateVector<glm::uvec2>("sintorn.tileSizeInPixels");
  //compute tile size in pixels
  tileSizeInPixels.resize(nofLevels,glm::uvec2(1u,1u));
  for(size_t l=0;l<nofLevels;++l)
    for(size_t m=l+1;m<nofLevels;++m)
      tileSizeInPixels[l]*=tileDivisibility[m];
}

void computeTileCount(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.tileDivisibility"}))return;

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels = tileDivisibility.size();

  auto&tileCount = vars.reCreateVector<glm::uvec2>("sintorn.tileCount");
  //compute level size
  tileCount.resize(nofLevels,glm::uvec2(1u,1u));
  std::cerr << "_tileCount.size: " << tileCount.size() << std::endl;
  std::cerr << "nofLevels: " << nofLevels << std::endl;
  for(size_t l=0;l<nofLevels;++l)
    for(size_t m=l;m<nofLevels;++m){
      auto const&td = tileDivisibility.at(l); 
      auto& tc = tileCount.at(m);
      tc *= td;
    }
}

#define ___ std::cerr << __FILE__ << ": " << __LINE__ << std::endl

void computeTileSizeInClipSpace(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.tileSizeInPixels","windowSize"}))return;

  auto const&tileSizeInPixels = vars.getVector<glm::uvec2>("sintorn.tileSizeInPixels");
  auto const windowSize       = *vars.get<glm::uvec2>("windowSize");

  auto&tileSizeInClipSpace = vars.reCreateVector<glm::vec2>("sintorn.tileSizeInClipSpace");

  //compute tiles sizes in clip space
  for(auto const&x:tileSizeInPixels)
    tileSizeInClipSpace.push_back(glm::vec2(2.f)/glm::vec2(windowSize)*glm::vec2(x));
}

void computeUsedTiles(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.tileDivisibility","windowSize"}))return;

  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();
  auto const windowSize       = *vars.get<glm::uvec2>("windowSize");

  auto&usedTiles = vars.reCreateVector<glm::uvec2>("sintorn.usedTiles");

  auto divRoundUp = [](uint32_t x,uint32_t y)->uint32_t{return (x/y)+((x%y)?1:0);};
  usedTiles.resize(nofLevels,glm::uvec2(0u,0u));
  usedTiles.back() = windowSize;
  for(int l=(int)nofLevels-2;l>=0;--l){
    usedTiles[l].x = divRoundUp(usedTiles[l+1].x,tileDivisibility[l+1].x);
    usedTiles[l].y = divRoundUp(usedTiles[l+1].y,tileDivisibility[l+1].y);
  }
}

void createWriteDepthProgram(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.tileDivisibility","sintorn.discardBackFacing"}))return;

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
  if(notChanged(vars,"sintorn",__FUNCTION__,{"wavefrontSize"}))return;

  auto wavefrontSize = vars.getSizeT("wavefrontSize");

  vars.reCreate<Program>("sintorn.hierarchicalDepthProgram",
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("DO_NOT_COUNT_WITH_INFINITY"                                                                     ),
        Shader::define("WAVEFRONT_SIZE"                            ,uint32_t(wavefrontSize                      )),
        Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT" ,int     (HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT )),
        Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT",int     (HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT)),
        sintorn::hierarchicalDepthSrc));

}

void allocateHierarchicalDepth(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"sintorn.usedTiles"}))return;

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

Sintorn::Sintorn(vars::Vars&vars):
  ShadowMethod(vars)
{
  assert(this!=nullptr);

  _shadowMask = vars.get<Texture>("shadowMask");

  _useUniformTileSizeInClipSpace=false;
  _useUniformTileDivisibility   =false;

  computeTileDivisibility(vars);
  computeTileSizeInPixel(vars);
  computeTileSizeInClipSpace(vars);
  computeTileCount(vars);
  computeUsedTiles(vars);

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
  

  createWriteDepthProgram(vars);

  //compile shader programs
#include<Sintorn/Shaders.h>

  auto const wavefrontSize = vars.getSizeT("wavefrontSize");
  createHierarchicalDepthProgram(vars);


  WriteStencilTextureProgram=make_shared<Program>(
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("LOCAL_TILE_SIZE_X"                           ,int(tileDivisibility[nofLevels-1].x)),
        Shader::define("LOCAL_TILE_SIZE_Y"                           ,int(tileDivisibility[nofLevels-1].y)),
        Shader::define("WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK",int(WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK )),
        Shader::define("WRITESTENCILTEXTURE_BINDING_HSTINPUT"        ,int(WRITESTENCILTEXTURE_BINDING_HSTINPUT         )),
        writeStencilTextureCompSrc));


  MergeTextureProgram=make_shared<Program>(
      make_shared<Shader>(
        GL_COMPUTE_SHADER,
        "#version 450 core\n",
        Shader::define("WAVEFRONT_SIZE"                ,uint32_t(wavefrontSize          )),
        Shader::define("MERGETEXTURE_BINDING_HSTINPUT" ,int     (MERGETEXTURE_BINDING_HSTINPUT )),
        Shader::define("MERGETEXTURE_BINDING_HSTOUTPUT",int     (MERGETEXTURE_BINDING_HSTOUTPUT)),
        mergeTextureCompSrc));

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

  _finalStencilMask = make_shared<Texture>(GL_TEXTURE_2D,GL_R32UI,1,vars.get<glm::uvec2>("windowSize")->x,vars.get<glm::uvec2>("windowSize")->y);
  _finalStencilMask->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
  _finalStencilMask->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST);

  allocateHierarchicalDepth(vars);

  size_t RESULT_LENGTH_IN_UINT=wavefrontSize/UINT_BIT_SIZE;
  if(RESULT_LENGTH_IN_UINT==0)RESULT_LENGTH_IN_UINT=1;

  for(size_t l=0;l<nofLevels;++l){
    _HST.push_back(make_shared<Texture>(GL_TEXTURE_2D,GL_R32UI,1,GLsizei(usedTiles[l].x*RESULT_LENGTH_IN_UINT),usedTiles[l].y));
    _HST.back()->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    _HST.back()->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST);
    uint8_t data[2] = {0,0};
    _HST.back()->clear(0,GL_RG_INTEGER,GL_UNSIGNED_BYTE,data);
    //glClearTexImage(_HDT.back()->getId(),0,GL_RG,GL_UNSIGNED_INT,&data);
  }
}

Sintorn::~Sintorn(){
}

void writeDepth(vars::Vars&vars,glm::vec4 const&lightPosition){
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

void Sintorn::GenerateHierarchyTexture(glm::vec4 const&lightPosition){
  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();
  auto const&tileSizeInPixels = vars.getVector<glm::uvec2>("sintorn.tileSizeInPixels");
  auto const&tileCount        = vars.getVector<glm::uvec2>("sintorn.tileCount");
  auto const&usedTiles        = vars.getVector<glm::uvec2>("sintorn.usedTiles");

  if(nofLevels<2)return;

  writeDepth(vars,lightPosition);

  auto HierarchicalDepthTextureProgram = vars.get<Program>("sintorn.hierarchicalDepthProgram");
  HierarchicalDepthTextureProgram->use();
  HierarchicalDepthTextureProgram->set2uiv("WindowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));

  HierarchicalDepthTextureProgram->set2uiv("TileDivisibility",glm::value_ptr(tileDivisibility.data()[0]),(GLsizei)nofLevels);
  HierarchicalDepthTextureProgram->set2uiv("TileSizeInPixels",glm::value_ptr(tileSizeInPixels.data()[0]),(GLsizei)nofLevels);

  auto&HDT = vars.getVector<shared_ptr<Texture>>("sintorn.HDT");
  for(int l=(int)nofLevels-2;l>=0;--l){
    HierarchicalDepthTextureProgram->set1ui("DstLevel",(unsigned)l);
    HDT[l+1]->bindImage(HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT );
    HDT[l  ]->bindImage(HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT);
    glDispatchCompute(usedTiles[l].x,usedTiles[l].y,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
  }
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

  _finalStencilMask->clear(0,GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
  //glClearTexImage(_finalStencilMask->getId(),0,GL_RED_INTEGER,GL_UNSIGNED_INT,NULL);
  for(size_t l=0;l<nofLevels;++l){
    _HST[l]->clear(0,GL_RED_INTEGER,GL_UNSIGNED_INT,nullptr);
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
    _HST[l]->bindImage(GLuint(RASTERIZETEXTURE_BINDING_HST+l));

  _finalStencilMask->bindImage(GLuint(RASTERIZETEXTURE_BINDING_FINALSTENCILMASK));

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

void Sintorn::MergeTexture(){
  auto const&tileDivisibility = vars.getVector<glm::uvec2>("sintorn.tileDivisibility");
  auto const nofLevels        = tileDivisibility.size();
  auto const&tileSizeInPixels = vars.getVector<glm::uvec2>("sintorn.tileSizeInPixels");
  auto const&tileCount        = vars.getVector<glm::uvec2>("sintorn.tileCount");

  MergeTextureProgram->use();
  MergeTextureProgram->set2uiv("WindowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));

  GLsync Sync=0;
  for(size_t l=0;l<nofLevels-1;++l){
    MergeTextureProgram->set2uiv("DstTileSizeInPixels",glm::value_ptr(tileSizeInPixels[l]));
    MergeTextureProgram->set2uiv("DstTileDivisibility",glm::value_ptr(tileDivisibility[l]));

    _HST[l  ]->bindImage(MERGETEXTURE_BINDING_HSTINPUT);
    _HST[l+1]->bindImage(MERGETEXTURE_BINDING_HSTOUTPUT);
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

  //glFinish();
  //glMemoryBarrier(GL_ALL_BARRIER_BITS);

  WriteStencilTextureProgram->use();
  WriteStencilTextureProgram->set2uiv("WindowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));

  _finalStencilMask->bindImage(WRITESTENCILTEXTURE_BINDING_FINALSTENCILMASK);
  _HST[nofLevels-1]->bindImage(WRITESTENCILTEXTURE_BINDING_HSTINPUT);

  glClientWaitSync(Sync,0,GL_TIMEOUT_IGNORED);
  glDeleteSync(Sync);

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
  GenerateHierarchyTexture(lightPosition);
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
  _HST[l]->bindImage(0);
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
  _finalStencilMask->bindImage(0);
  _emptyVao->bind();
  glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  _emptyVao->unbind();
}

void Sintorn::blit(){
  assert(this!=nullptr);
  assert(_blitProgram!=nullptr);
  assert(_finalStencilMask!=nullptr);
  assert(_shadowMask!=nullptr);
  _blitProgram->use();
  _finalStencilMask->bindImage(0);
  _shadowMask->bindImage(1);
  _blitProgram->set2uiv("windowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));
  glDispatchCompute(
      (GLuint)getDispatchSize(vars.get<glm::uvec2>("windowSize")->x,8),
      (GLuint)getDispatchSize(vars.get<glm::uvec2>("windowSize")->y,8),1);
}
