#include<RSSV/RSSV.h>

#include<ProgramExtension.h>
#include<util.h>
#include<Deferred.h>
#include<ShadowVolumes.h>
#include<RSSV/BuildStupidHierarchy.h>
#include<RSSV/PerfectResolution/Build.h>

using namespace rssv;

const size_t HIERARCHICALDEPTHTEXTURE_BINDING_DEPTH     = 0;
const size_t HIERARCHICALDEPTHTEXTURE_BINDING_HDT       = 1;

const size_t HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT  = 0;
const size_t HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT = 1;

const size_t RASTERIZE_BINDING_HDT         = 0;
const size_t RASTERIZE_BINDING_SSM         = 1;
const size_t RASTERIZE_BINDING_SILHOUETTES = 0;

const size_t     WAVEFRONT_SIZE = 64;
const size_t     TILE_EDGE_SIZE = size_t(glm::sqrt(WAVEFRONT_SIZE));
const glm::uvec2 TILE_SIZE      = glm::uvec2(uint32_t(TILE_EDGE_SIZE),uint32_t(TILE_EDGE_SIZE));

void RSSV::_allocateHDT(){
  auto const nofLevels = _tiling.borderTileDivisibilityIntoFullTiles.size();
  for(size_t level=0;level<nofLevels;++level){
    auto size = _tiling.hdtSize.at(level);
    _HDT.push_back(
        std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_RG32F,1,size.x,size.y));
  }
}

RSSV::RSSV(vars::Vars&vars           ):
  ShadowMethod(vars),
  _tiling      (*vars.get<glm::uvec2>("windowSize"),vars.getSizeT("wavefrontSize"))
{
  buildHierarchy = std::make_shared<BuildStupidHierarchy>(vars);
  //buildHierarchy = std::make_shared<PerfectHierarchy>(vars);
  //extractSilhouettes = std::make_shared<RSSVExtractSilhouettes>(vars);
#if 0
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  assert(this!=nullptr);
  assert(
      (windowSize.x==8    && windowSize.y==8   ) ||
      (windowSize.x==64   && windowSize.y==64  ) ||
      (windowSize.x==512  && windowSize.y==512 ) ||
      (windowSize.x==4096 && windowSize.y==4096) );
  if(windowSize.x == 8   )this->_nofLevels = 1;
  if(windowSize.x == 64  )this->_nofLevels = 2;
  if(windowSize.x == 512 )this->_nofLevels = 3;
  if(windowSize.x == 4096)this->_nofLevels = 4;

  _allocateHDT();



#include"SilhouetteShaders.h"
#include"RSSVShaders.h"

  this->_generateHDT0Program = makeComputeProgram(
      defineComputeShaderHeader(TILE_SIZE,WAVEFRONT_SIZE),
      ge::gl::Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_DEPTH",(int)HIERARCHICALDEPTHTEXTURE_BINDING_DEPTH),
      ge::gl::Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_HDT"  ,(int)HIERARCHICALDEPTHTEXTURE_BINDING_HDT  ),
      _generateHDT0Src);
  this->_generateHDTProgram = makeComputeProgram(
      defineComputeShaderHeader(TILE_SIZE,WAVEFRONT_SIZE),
      ge::gl::Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT" ,(int)HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT ),
      ge::gl::Shader::define("HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT",(int)HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT),
      _generateHDTSrc);
  glm::uvec2 hdtSize = TILE_SIZE;
  for(size_t l=0;l<this->_nofLevels;++l){
    this->_HDT.push_back(std::make_shared<ge::gl::Texture>(GL_TEXTURE_2D,GL_RG32F,1,hdtSize.x,hdtSize.y));
    this->_HDT.back()->texParameteri(GL_TEXTURE_MAG_FILTER,GL_NEAREST               );
    this->_HDT.back()->texParameteri(GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST);
    float data[2]={1,-1};
    this->_HDT.back()->clear(0,GL_RG,GL_FLOAT,data);
    hdtSize*=TILE_SIZE;
  }


  this->_computeSilhouettesProgram = makeComputeProgram(
      defineComputeShaderHeader(vars.getSizeT("rssv.computeSilhouetteWGS"),WAVEFRONT_SIZE),
      ge::gl::Shader::define("MAX_MULTIPLICITY",int(vars.getSizeT("maxMultiplicity"))),
      ge::gl::Shader::define("LOCAL_ATOMIC"    ,int(vars.getBool("rssv.localAtomic"))),
      ge::gl::Shader::define("CULL_SIDES"      ,int(vars.getBool("rssv.cullSides"  ))),
      ge::gl::Shader::define("USE_PLANES"      ,int(vars.getBool("rssv.usePlanes"  ))),
      silhouetteFunctions,
      _computeSilhouettesSrc);

  auto adj = createAdjacency(vars);

  size_t const nofVec4PerEdge = verticesPerEdge + adj->getMaxMultiplicity();
  this->_edges = std::make_shared<ge::gl::Buffer>(sizeof(float)*componentsPerVertex4D*nofVec4PerEdge*adj->getNofEdges());

  float      *dstPtr = (float      *)this->_edges->map();
  auto const srcPtr = adj->getVertices().data();

  for(size_t e=0;e<adj->getNofEdges();++e){
    auto dstEdgePtr             = dstPtr + e*nofVec4PerEdge*componentsPerVertex4D;
    auto dstVertexAPtr          = dstEdgePtr;
    auto dstVertexBPtr          = dstVertexAPtr + componentsPerVertex4D;
    auto dstOppositeVerticesPtr = dstVertexBPtr + componentsPerVertex4D;

    auto srcVertexAPtr          = srcPtr + adj->getEdge(e,0);
    auto srcVertexBPtr          = srcPtr + adj->getEdge(e,1);

    size_t const sizeofVertex3DInBytes = componentsPerVertex3D * sizeof(float);

    std::memcpy(dstVertexAPtr,srcVertexAPtr,sizeofVertex3DInBytes);
    dstVertexAPtr[3] = (float)adj->getNofOpposite(e);

    std::memcpy(dstVertexBPtr,srcVertexBPtr,sizeofVertex3DInBytes);
    dstVertexBPtr[3] = 1.f;

    for(size_t o=0;o<adj->getNofOpposite(e);++o){
      auto dstOppositeVertexPtr = dstOppositeVerticesPtr + o*componentsPerVertex4D;
      if(vars.getBool("rssv.usePlanes")){
        auto const plane = computePlane(toVec3(srcPtr+adj->getEdgeVertexA(e)),toVec3(srcPtr+adj->getEdgeVertexB(e)),toVec3(srcPtr+adj->getOpposite(e,o)));
        std::memcpy(dstOppositeVertexPtr,&plane,sizeof(plane));
      }else{
        auto srcOppositeVertexPtr = srcPtr + adj->getOpposite(e,o);
        std::memcpy(dstOppositeVertexPtr,srcOppositeVertexPtr,sizeofVertex3DInBytes);
        dstOppositeVertexPtr[3] = 1.f;
      }
    }

    size_t const nofEmptyOppositeVertices = adj->getMaxMultiplicity() - adj->getNofOpposite(e);
    size_t const sizeofEmptyVerticesInBytes = sizeof(float)*componentsPerVertex4D*nofEmptyOppositeVertices;
    auto dstEmptyOppositeVerticesPtr = dstOppositeVerticesPtr + adj->getNofOpposite(e)*componentsPerVertex4D;
    std::memset(dstEmptyOppositeVerticesPtr,0,sizeofEmptyVerticesInBytes);
  }
  this->_edges->unmap();
  this->_nofEdges = adj->getNofEdges();

  this->_silhouettes=std::make_shared<ge::gl::Buffer>(
      sizeof(float)*componentsPerVertex4D*2*this->_nofEdges,
      nullptr,GL_DYNAMIC_COPY);
  this->_silhouettes->clear(GL_R32F,GL_RED,GL_FLOAT);

  struct DispatchIndirectCommand{
    uint32_t nofWorkGroupsX = 0;
    uint32_t nofWorkGroupsY = 0;
    uint32_t nofWorkGroupsZ = 0;
  };
  DispatchIndirectCommand cmd;
  cmd.nofWorkGroupsY = 1;
  cmd.nofWorkGroupsZ = 1;
  this->_dispatchIndirectBuffer=std::make_shared<ge::gl::Buffer>(sizeof(DispatchIndirectCommand),&cmd);

  this->_rasterizeProgram = makeComputeProgram(
      defineComputeShaderHeader(glm::uvec2(WAVEFRONT_SIZE,vars.getSizeT("rssv.silhouettesPerWorkgroup")),WAVEFRONT_SIZE),
      ge::gl::Shader::define("NUMBER_OF_LEVELS"             ,(int)this->_nofLevels               ),
      ge::gl::Shader::define("NUMBER_OF_LEVELS_MINUS_ONE"   ,(int)this->_nofLevels-1             ),
      ge::gl::Shader::define("SILHOUETTES_PER_WORKGROUP"    ,(int)vars.getSizeT("rssv.silhouettesPerWorkgroup") ),
      ge::gl::Shader::define("RASTERIZE_BINDING_SSM"        ,(int)RASTERIZE_BINDING_SSM          ),
      ge::gl::Shader::define("RASTERIZE_BINDING_HDT"        ,(int)RASTERIZE_BINDING_HDT          ),
      ge::gl::Shader::define("RASTERIZE_BINDING_SILHOUETTES",(int)RASTERIZE_BINDING_SILHOUETTES  ),
      _rasterizeSrc);

#endif
}

RSSV::~RSSV(){}

void RSSV::create(
    glm::vec4 const&lightPosition,
    glm::mat4 const&view         ,
    glm::mat4 const&projection   ){
  buildHierarchy->build();
  ifExistStamp("buildHierarchy");
  //extractSilhouettes->extract(lightPosition);
  //ifExistStamp("extractSilhouettes");
#if 0
  assert(this!=nullptr);
  (void)lightPosition;
  (void)view;
  (void)projection;
  ifExistStamp("");
  this->_generateHDT();
  ifExistStamp("computeHDT");
  this->_computeSilhouettes(lightPosition);
  ifExistStamp("computeSilhouettes");
  this->_rasterize(lightPosition,view,projection);
  ifExistStamp("rasterize");
#endif
}

void RSSV::_copyDepthToLastLevelOfHDT(){
#if 0
  assert(this!=nullptr);
  this->_generateHDT0Program->use();
  this->_generateHDT0Program->set2uiv("windowSize",glm::value_ptr(*vars.get<glm::uvec2>("windowSize")));
  vars.get<GBuffer>("gBuffer")->depth->bind(HIERARCHICALDEPTHTEXTURE_BINDING_DEPTH);
  this->_HDT.back()->bindImage(HIERARCHICALDEPTHTEXTURE_BINDING_HDT);
  glm::uvec2 nofWorkGroups = divRoundUp(_tiling.hdtSize.back(),*vars.get<glm::uvec2>("rssv.copyDepthToLastLevelOfHDTWGS"));
    
  glDispatchCompute(nofWorkGroups.x,nofWorkGroups.y,1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
#endif
}

void RSSV::_computeAllLevelsOfHDTExceptLast(){
#if 0
  this->_generateHDTProgram->use();

  for(int64_t l=_nofLevels-2;l>=0;--l){
    _generateHDTProgram->set2uiv("inputSize",glm::value_ptr(_tiling.hdtSize.at(l+1)));
    _HDT[l+1]->bindImage(HIERARCHICALDEPTHTEXTURE_BINDING_HDTINPUT );
    _HDT[l  ]->bindImage(HIERARCHICALDEPTHTEXTURE_BINDING_HDTOUTPUT);
    glm::uvec2 nofWorkGroups = _tiling.hdtSize.at(l);
    glDispatchCompute(nofWorkGroups.x,nofWorkGroups.y,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
  }
#endif
}

void RSSV::_generateHDT(){
#if 0
  _copyDepthToLastLevelOfHDT();
  _computeAllLevelsOfHDTExceptLast();
#endif
}

void RSSV::_computeSilhouettes(glm::vec4 const&lightPosition){
#if 0
  assert(this!=nullptr);
  this->_dispatchIndirectBuffer->clear(GL_R32UI,0,sizeof(unsigned),GL_RED_INTEGER,GL_UNSIGNED_INT);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  this->_computeSilhouettesProgram
    ->set1ui    ("numEdge"               ,uint32_t(this->_nofEdges)    )
    ->set4fv    ("lightPosition"         ,glm::value_ptr(lightPosition))
    ->bindBuffer("edges"                 ,this->_edges                 )
    ->bindBuffer("silhouettes"           ,this->_silhouettes           )
    ->bindBuffer("dispatchIndirectBuffer",this->_dispatchIndirectBuffer)
    ->dispatch(GLuint(getDispatchSize(this->_nofEdges,vars.getSizeT("rssv.computeSilhouetteWGS"))));
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  /*
  auto ptr = (float*)this->_silhouettes->map();
  for(size_t i=0;i<10;++i){
    std::cout<<ptr[i*7+0]<<" "<<ptr[i*7+1]<<" "<<ptr[i*7+2]<<std::endl;
    std::cout<<ptr[i*7+3]<<" "<<ptr[i*7+4]<<" "<<ptr[i*7+5]<<std::endl;
    std::cout<<ptr[i*7+6]<<std::endl;
  }
  this->_silhouettes->unmap();
  // */
  /*
  auto ptr = (float*)this->_edges->map();
  for(size_t i=0;i<10;++i){
    for(size_t k=0;k<4;++k){
      for(size_t l=0;l<4;++l)
        std::cout<<ptr[(i*4+k)*4+l]<<" ";
      std::cout<<std::endl;
    }
    std::cout<<std::endl;
  }
  this->_edges->unmap();
  // */
#endif
}

void RSSV::_rasterize(glm::vec4 const&lightPosition,glm::mat4 const&view,glm::mat4 const&projection){
  assert(this!=nullptr);
  (void)lightPosition;
  (void)view;
  (void)projection;
}

