#include <Sintorn/Rasterize.h>
#include <Sintorn/RasterizationShaders.h>
#include <Barrier.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <Deferred.h>
#include <util.h>
#include <sstream>
#include <Vars/Caller.h>

using namespace std;
using namespace ge::gl;

size_t RASTERIZETEXTURE_BINDING_FINALSTENCILMASK = 0;
size_t RASTERIZETEXTURE_BINDING_HST              = 1;
size_t RASTERIZETEXTURE_BINDING_HDT              = 5;
size_t RASTERIZETEXTURE_BINDING_TRIANGLE_ID      = 9;
size_t RASTERIZETEXTURE_BINDING_SHADOWFRUSTA     = 0;

void createRasterizationProgram(vars::Vars&vars){
  if(notChanged(vars,"sintorn",__FUNCTION__,{"wavefrontSize","sintorn.tileDivisibility","sintorn.tileSizeInClipSpace","sintorn.useUniformTileDivisibility","sintorn.useUniformTileSizeInClipSpace","args.sintorn.shadowFrustaPerWorkGroup"}))return;
  vars::Caller caller(vars,__FUNCTION__);

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
        Shader::define("SHADOWFRUSTUMS_PER_WORKGROUP",int(vars.getUint32("args.sintorn.shadowFrustaPerWorkGroup"))),
        TileSizeInClipSpaceDefines,
        TileDivisibilityDefines,
        Shader::define("RASTERIZETEXTURE_BINDING_FINALSTENCILMASK",int(RASTERIZETEXTURE_BINDING_FINALSTENCILMASK)),
        Shader::define("RASTERIZETEXTURE_BINDING_HST"             ,int(RASTERIZETEXTURE_BINDING_HST             )),
        Shader::define("RASTERIZETEXTURE_BINDING_HDT"             ,int(RASTERIZETEXTURE_BINDING_HDT             )),
        Shader::define("RASTERIZETEXTURE_BINDING_TRIANGLE_ID"     ,int(RASTERIZETEXTURE_BINDING_TRIANGLE_ID     )),
        Shader::define("RASTERIZETEXTURE_BINDING_SHADOWFRUSTA"    ,int(RASTERIZETEXTURE_BINDING_SHADOWFRUSTA    )),
        sintorn::rasterizationShader));
}

void rasterize(vars::Vars&vars){
  vars::Caller caller(vars,__FUNCTION__);
  createRasterizationProgram(vars);

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
  size_t workgroups = getDispatchSize(vars.getSizeT("sintorn.nofTriangles"),vars.getUint32("args.sintorn.shadowFrustaPerWorkGroup"));
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

