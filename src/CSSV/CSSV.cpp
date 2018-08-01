#include<CSSV/CSSV.h>
#include<FastAdjacency.h>
#include<util.h>
#include<geGL/StaticCalls.h>

#include<CSSV/BasicExtractSilhouettes.h>
#include<CSSV/PlanesExtractSilhouettes.h>
#include<CSSV/InterleavedPlanesExtractSilhouettes.h>

using namespace cssv;

std::shared_ptr<ge::gl::VertexArray>createSidesVao(std::shared_ptr<ge::gl::Buffer>const&sillhouettes){
  auto sidesVao = std::make_shared<ge::gl::VertexArray>();
  sidesVao->addAttrib(sillhouettes,0,componentsPerVertex4D,GL_FLOAT);
  return sidesVao;
}

void CSSV::_createCapsData (std::shared_ptr<Adjacency const>const&adj){
  assert(this!=nullptr);
  assert(adj!=nullptr);
  auto const nofTriangles = adj->getNofTriangles();
  std::vector<float>dst(componentsPerVertex4D*verticesPerTriangle*nofTriangles);

  auto const dstPtr = dst.data();
  auto const srcPtr = static_cast<float const*>(adj->getVertices());
  for(size_t t=0;t<nofTriangles;++t){
    auto const triangleDstPtr = dstPtr + t*componentsPerVertex4D*verticesPerTriangle;
    auto const triangleSrcPtr = srcPtr + t*componentsPerVertex3D*verticesPerTriangle;
    for(size_t p=0;p<verticesPerTriangle;++p){
      auto   const vertexDstPtr = triangleDstPtr + p*componentsPerVertex4D;
      auto   const vertexSrcPtr = triangleSrcPtr + p*componentsPerVertex3D;
      size_t const sizeofVertex3DInBytes = componentsPerVertex3D * sizeof(float);
      std::memcpy(vertexDstPtr,vertexSrcPtr,sizeofVertex3DInBytes);
      vertexDstPtr[3] = 1.f;
    }
  }
  _caps    = std::make_shared<ge::gl::Buffer>(dst);
  _capsVao = std::make_shared<ge::gl::VertexArray>();
  _capsVao->addAttrib(_caps,0,4,GL_FLOAT);
}



CSSV::CSSV(vars::Vars&vars):
  ShadowVolumes(vars  )
{
  assert(this!=nullptr);

  std::vector<float>vertices;
  vars.get<Model>("model")->getVertices(vertices);

  size_t const nofTriangles = vertices.size() / (verticesPerTriangle*componentsPerVertex3D);
  auto const adj = std::make_shared<Adjacency const>(vertices.data(),nofTriangles,vars.getSizeT("maxMultiplicity"));

  _nofTriangles=adj->getNofTriangles();

  if(vars.getBool("cssv.usePlanes")){
    if(vars.getBool("cssv.useInterleaving"))
      extractSilhouettes = std::make_unique<InterleavedPlanesExtractSilhouettes>(vars,adj);
    else
      extractSilhouettes = std::make_unique<PlanesExtractSilhouettes>(vars,adj);
  }else
    extractSilhouettes = std::make_unique<BasicExtractSilhouettes>(vars,adj);

  _sidesVao = createSidesVao(extractSilhouettes->sillhouettes);

  _createCapsData(adj);

  struct DrawArraysIndirectCommand{
    uint32_t nofVertices  = 0;
    uint32_t nofInstances = 0;
    uint32_t firstVertex  = 0;
    uint32_t baseInstance = 0;
  };
  DrawArraysIndirectCommand cmd;
  cmd.nofInstances = 1;
  extractSilhouettes->dibo=std::make_shared<ge::gl::Buffer>(sizeof(DrawArraysIndirectCommand),&cmd);

#include<CSSV/Shaders.h>
#include<SilhouetteShaders.h>


  _drawSidesProgram=std::make_shared<ge::gl::Program>(
      std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER         ,drawVPSrc),
      std::make_shared<ge::gl::Shader>(GL_TESS_CONTROL_SHADER   ,drawCPSrc),
      std::make_shared<ge::gl::Shader>(GL_TESS_EVALUATION_SHADER,drawEPSrc));

  _drawCapsProgram = std::make_shared<ge::gl::Program>(
      std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER  ,capsVPSrc),
      std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        capsGPSrc));
}

CSSV::~CSSV(){}

void CSSV::drawSides(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){

  extractSilhouettes->compute(lightPosition);

  ifExistStamp("compute");

  _sidesVao->bind();
  auto mvp = projectionMatrix * viewMatrix;

  _drawSidesProgram
    ->setMatrix4fv("mvp"          ,glm::value_ptr(mvp          ))
    ->set4fv      ("lightPosition",glm::value_ptr(lightPosition));
  extractSilhouettes->dibo->bind(GL_DRAW_INDIRECT_BUFFER);
  glPatchParameteri(GL_PATCH_VERTICES,2);
  _drawSidesProgram->use();
  glDrawArraysIndirect(GL_PATCHES,NULL);
  _sidesVao->unbind();
}

void CSSV::drawCaps(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  auto mvp = projectionMatrix * viewMatrix;
  _drawCapsProgram
    ->setMatrix4fv("mvp"          ,glm::value_ptr(mvp          ))
    ->set4fv      ("lightPosition",glm::value_ptr(lightPosition));
  _drawCapsProgram->use();
  _capsVao->bind();
  glDrawArrays(GL_TRIANGLES,0,(GLsizei)_nofTriangles*3);
  _capsVao->unbind();
}

