#include<VSSV/VSSV.h>
#include<FastAdjacency.h>
#include<VSSV/Shaders.h>
#include<Simplex.h>
#include<VSSV/DrawSidesUsingPoints.h>

/*
size_t const floatsPerNofOppositeVertices = 1;
size_t const sizeofVertex3DInBytes        = componentsPerVertex3D  *sizeof(float);
size_t const sizeofPlane3DInBytes         = componentsPerPlane3D   *sizeof(float);

void VSSV::createSideDataUsingAllPlanes(std::shared_ptr<Adjacency const>const&adj){
  assert(this!=nullptr);
  assert(adj!=nullptr);
  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  //(A,B,M,T0,T1,...)*
  //A - vertex A of an edge 3*float
  //B - vertex B of an edge 3*float
  //Tn - triangle planes n*4*float
  //n < maxMultiplicity
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + maxNofOppositeVertices*componentsPerPlane3D;
  adjacency = std::make_shared<ge::gl::Buffer>(adj->getNofEdges()*floatsPerEdge*sizeof(float));
  float*ptr=(float*)adjacency->map();
  for(size_t edgeIndex=0;edgeIndex<adj->getNofEdges();++edgeIndex){
    auto edgePtr                 = ptr + edgeIndex*floatsPerEdge;
    auto edgeVertexAPtr          = edgePtr;
    auto edgeVertexBPtr          = edgeVertexAPtr + componentsPerVertex3D;
    auto edgeOppositeVerticesPtr = edgeVertexBPtr + componentsPerVertex3D;
    size_t const edgeVertexAIndex = 0;
    size_t const edgeVertexBIndex = 1;
    std::memcpy(edgeVertexAPtr,adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexAIndex),sizeofVertex3DInBytes);
    std::memcpy(edgeVertexBPtr,adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexBIndex),sizeofVertex3DInBytes);
    for(size_t oppositeIndex=0;oppositeIndex<adj->getNofOpposite(edgeIndex);++oppositeIndex)
      std::memcpy(
          edgeOppositeVerticesPtr+oppositeIndex*componentsPerPlane3D,
          glm::value_ptr(computePlane(
              toVec3(adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexAIndex) ),
              toVec3(adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexBIndex) ),
              toVec3(adj->getVertices().data()+adj->getOpposite(edgeIndex,oppositeIndex))
              )),
          sizeofPlane3DInBytes);
    size_t const nofEmptyOppositeVertices = maxNofOppositeVertices-adj->getNofOpposite(edgeIndex);
    std::memset(
        edgeOppositeVerticesPtr+adj->getNofOpposite(edgeIndex)*componentsPerPlane3D,
        0,
        sizeofPlane3DInBytes*nofEmptyOppositeVertices);
  }
  adjacency->unmap();
  nofEdges = adj->getNofEdges();

  //create vertex array for sides
  //divisor = maxMultiplicity -> attrib are modified once per edge
  sidesVao = std::make_shared<ge::gl::VertexArray>();
  GLenum const normalized = GL_FALSE;
  GLuint const divisor = GLuint(adj->getMaxMultiplicity());
  GLintptr offset = 0;
  GLsizei const stride = GLsizei(floatsPerEdge*sizeof(float));
  sidesVao->addAttrib(adjacency,0,componentsPerVertex3D ,GL_FLOAT,stride,offset,normalized,divisor);
  offset += componentsPerVertex3D*sizeof(float);
  sidesVao->addAttrib(adjacency,1,componentsPerVertex3D ,GL_FLOAT,stride,offset,normalized,divisor);
  offset += componentsPerVertex3D*sizeof(float);
  for(GLuint oppositeIndex=0;oppositeIndex<adj->getMaxMultiplicity();++oppositeIndex){
    sidesVao->addAttrib(adjacency,3+oppositeIndex,componentsPerPlane3D,GL_FLOAT,stride,offset,normalized,divisor);
    offset += componentsPerPlane3D*sizeof(float);
  }
}

void VSSV::createSideDataUsingPlanes(std::shared_ptr<Adjacency const>const&adj){
  assert(this!=nullptr);
  assert(adj!=nullptr);
  size_t const maxNofOppositeVertices = adj->getMaxMultiplicity();
  //(A,B,M,T0,T1,...)*
  //A - vertex A of an edge 3*float
  //B - vertex B of an edge 3*float
  //M - multiplicity of an edge 1*float
  //Tn - triangle planes n*4*float
  //n < maxMultiplicity
  size_t const floatsPerEdge = verticesPerEdge*componentsPerVertex3D + floatsPerNofOppositeVertices + maxNofOppositeVertices*componentsPerPlane3D;
  adjacency = std::make_shared<ge::gl::Buffer>(adj->getNofEdges()*floatsPerEdge*sizeof(float));
  float*ptr=(float*)adjacency->map();
  for(size_t edgeIndex=0;edgeIndex<adj->getNofEdges();++edgeIndex){
    auto edgePtr = ptr+edgeIndex*floatsPerEdge;
    auto edgeVertexAPtr = edgePtr;
    auto edgeVertexBPtr = edgeVertexAPtr + componentsPerVertex3D;
    auto edgeNofOppositePtr = edgeVertexBPtr + componentsPerVertex3D;
    auto edgeOppositeVerticesPtr = edgeNofOppositePtr + floatsPerNofOppositeVertices;
    size_t const edgeVertexAIndex = 0;
    size_t const edgeVertexBIndex = 1;
    std::memcpy(edgeVertexAPtr,adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexAIndex),sizeofVertex3DInBytes);
    std::memcpy(edgeVertexBPtr,adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexBIndex),sizeofVertex3DInBytes);
    *edgeNofOppositePtr = float(adj->getNofOpposite(edgeIndex));
    for(size_t oppositeIndex=0;oppositeIndex<adj->getNofOpposite(edgeIndex);++oppositeIndex)
      std::memcpy(
          edgeOppositeVerticesPtr+oppositeIndex*componentsPerPlane3D,
          glm::value_ptr(computePlane(
              toVec3(adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexAIndex) ),
              toVec3(adj->getVertices().data()+adj->getEdge(edgeIndex,edgeVertexBIndex) ),
              toVec3(adj->getVertices().data()+adj->getOpposite(edgeIndex,oppositeIndex))
              )),
          sizeofPlane3DInBytes);
    size_t const nofEmptyOppositeVertices = maxNofOppositeVertices-adj->getNofOpposite(edgeIndex);
    std::memset(
        edgeOppositeVerticesPtr+adj->getNofOpposite(edgeIndex)*componentsPerPlane3D,
        0,
        sizeofPlane3DInBytes*nofEmptyOppositeVertices);
  }
  adjacency->unmap();
  nofEdges = adj->getNofEdges();

  //create vertex array for sides
  //divisor = maxMultiplicity -> attrib are modified once per edge
  sidesVao = std::make_shared<ge::gl::VertexArray>();
  GLenum const normalized = GL_FALSE;
  GLuint const divisor = GLuint(adj->getMaxMultiplicity());
  GLintptr offset = 0;
  GLsizei const stride = GLsizei(floatsPerEdge*sizeof(float));
  sidesVao->addAttrib(adjacency,0,componentsPerVertex3D ,GL_FLOAT,stride,offset,normalized,divisor);
  offset += componentsPerVertex3D*sizeof(float);
  sidesVao->addAttrib(adjacency,1,componentsPerVertex3D ,GL_FLOAT,stride,offset,normalized,divisor);
  offset += componentsPerVertex3D*sizeof(float);
  sidesVao->addAttrib(adjacency,2,floatsPerNofOppositeVertices,GL_FLOAT,stride,offset,normalized,divisor);
  offset += floatsPerNofOppositeVertices*sizeof(float);
  for(GLuint oppositeIndex=0;oppositeIndex<adj->getMaxMultiplicity();++oppositeIndex){
    sidesVao->addAttrib(adjacency,3+oppositeIndex,componentsPerPlane3D,GL_FLOAT,stride,offset,normalized,divisor);
    offset += componentsPerPlane3D*sizeof(float);
  }
}
*/
VSSV::VSSV(vars::Vars&vars):
  ShadowVolumes(vars       )
{
  assert(this!=nullptr);

  //compute adjacency of the model
  auto const adj = createAdjacency(vars);
  /*
  maxMultiplicity = adj->getMaxMultiplicity();

  //create and fill adjacency buffer for sides on GPU
  if(vars.getBool("vssv.usePlanes")){
    if(vars.getBool("vssv.useAllOppositeVertices")){
      createSideDataUsingAllPlanes(adj);
    }else{
      createSideDataUsingPlanes(adj);
    }
  }else{
    createSideDataUsingPoints(adj);
  }

#include"VSSV/Shaders.h"
#include"SilhouetteShaders.h"

  drawSidesProgram = std::make_shared<ge::gl::Program>(
      std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        vars.getBool("vssv.usePlanes"             )?ge::gl::Shader::define("USE_PLANES"               ):"",
        vars.getBool("vssv.useStrips"             )?ge::gl::Shader::define("USE_TRIANGLE_STRIPS"      ):"",
        vars.getBool("vssv.useAllOppositeVertices")?ge::gl::Shader::define("USE_ALL_OPPOSITE_VERTICES"):"",
        silhouetteFunctions,
        _drawSidesVertexShaderSrc));
*/
  caps = std::make_unique<DrawCaps>(adj);
  sides = std::make_unique<DrawSidesUsingPoints>(vars,adj);
}

VSSV::~VSSV(){}

void VSSV::drawSides(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  sides->draw(lightPosition,viewMatrix,projectionMatrix);
  /*
  assert(this!=nullptr);
  assert(drawSidesProgram!=nullptr);
  assert(sidesVao!=nullptr);
  drawSidesProgram->use();
  drawSidesProgram->setMatrix4fv("viewMatrix"      ,glm::value_ptr(viewMatrix      ));
  drawSidesProgram->setMatrix4fv("projectionMatrix",glm::value_ptr(projectionMatrix));
  drawSidesProgram->set4fv      ("lightPosition"   ,glm::value_ptr(lightPosition   ));
  sidesVao->bind();
  if(vars.getBool("vssv.useStrips"))
    glDrawArraysInstanced(GL_TRIANGLE_STRIP,0,4,GLsizei(nofEdges*maxMultiplicity));
  else
    glDrawArraysInstanced(GL_TRIANGLES     ,0,6,GLsizei(nofEdges*maxMultiplicity));
  sidesVao->unbind();
  */
}

void VSSV::drawCaps(
    glm::vec4 const&lightPosition,
    glm::mat4 const&viewMatrix         ,
    glm::mat4 const&projectionMatrix   ){
  caps->draw(lightPosition,viewMatrix,projectionMatrix);
}

