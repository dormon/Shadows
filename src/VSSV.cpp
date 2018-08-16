#include<VSSV.h>
#include<FastAdjacency.h>
#include<VSSVShaders.h>
#include<Simplex.h>

size_t const floatsPerNofOppositeVertices = 1;
size_t const sizeofVertex3DInBytes        = componentsPerVertex3D  *sizeof(float);
size_t const sizeofPlane3DInBytes         = componentsPerPlane3D   *sizeof(float);

template<size_t N=2>
struct GPUEdgeData{
  Vertex3Df vertexA;
  Vertex3Df vertexB;
  float     nofOpposite;
  Vertex3Df oppositeVertices[N];
};

template<size_t N=2>
void writeEdge(GPUEdgeData<N>&edge,Vertex3Df const*const vertices,size_t e,std::shared_ptr<Adjacency const>const&adj){
  edge.vertexA     = vertices[adj->getEdgeVertexA(e)/3];
  edge.vertexB     = vertices[adj->getEdgeVertexB(e)/3];
  edge.nofOpposite = adj->getNofOpposite(e);
  for(size_t o=0;o<adj->getNofOpposite(e);++o)
    edge.oppositeVertices[o] = vertices[adj->getOpposite(e,o)/3];
  for(size_t o=adj->getNofOpposite(e);o<2;++o)
    edge.oppositeVertices[o].clear();
}

template<size_t N=2>
void writeEdges(std::vector<GPUEdgeData<2>>&dst,Vertex3Df const*const src,std::shared_ptr<Adjacency const>const&adj){
  for(size_t e=0;e<adj->getNofEdges();++e)
    writeEdge(dst[e],src,e,adj);
}

void VSSV::createSideDataUsingPoints(std::shared_ptr<Adjacency const>const&adj){
  assert(this!=nullptr);
  assert(adj!=nullptr);
  std::vector<GPUEdgeData<2>>dst(adj->getNofEdges());
  auto const src = reinterpret_cast<Vertex3Df const*>(adj->getVertices().data());

  writeEdges(dst,src,adj);
  adjacency = std::make_shared<ge::gl::Buffer>(dst);
  nofEdges = adj->getNofEdges();

  //create vertex array for sides
  //divisor = maxMultiplicity -> attrib are modified once per edge
  sidesVao = std::make_shared<ge::gl::VertexArray>();
  GLenum const normalized = GL_FALSE;
  GLuint const divisor = GLuint(adj->getMaxMultiplicity());
  GLsizei const stride = GLsizei(sizeof(GPUEdgeData<2>));
  sidesVao->addAttrib(adjacency,0,sizeof(GPUEdgeData<2>::vertexA    )/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<2>,vertexA    ),normalized,divisor);
  sidesVao->addAttrib(adjacency,1,sizeof(GPUEdgeData<2>::vertexB    )/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<2>,vertexB    ),normalized,divisor);
  sidesVao->addAttrib(adjacency,2,sizeof(GPUEdgeData<2>::nofOpposite)/sizeof(float),GL_FLOAT,stride,offsetof(GPUEdgeData<2>,nofOpposite),normalized,divisor);
  for(GLuint o=0;o<adj->getMaxMultiplicity();++o){
    sidesVao->addAttrib(adjacency,3+o,componentsPerVertex3D,GL_FLOAT,stride,offsetof(GPUEdgeData<2>,oppositeVertices)+o*sizeof(GPUEdgeData<2>::oppositeVertices[0]),normalized,divisor);
  }
}

void VSSV::createCapDataUsingPoints(std::shared_ptr<Adjacency const>const&adj){
  assert(this!=nullptr);
  assert(adj!=nullptr);
  //create and fill adjacency buffer on GPU
  //(A,B,C)*
  //A - vertex A of an triangle
  //B - vertex B of an triangle
  //C - vertex C of an triangle

  size_t const sizeofTriangleInBytes = componentsPerVertex3D*verticesPerTriangle*sizeof(float);
  caps = std::make_shared<ge::gl::Buffer>(adj->getVertices());

  capsVao = std::make_shared<ge::gl::VertexArray>();
  GLsizei const stride     = GLsizei(sizeofTriangleInBytes);
  GLenum  const normalized = GL_FALSE;
  size_t  const nofCapsPerTriangle = 2;
  GLuint  const divisor    = GLuint(nofCapsPerTriangle);
  for(size_t i=0;i<verticesPerTriangle;++i){
    GLintptr offset = sizeofVertex3DInBytes * i;
    GLuint   index = GLuint(i);
    capsVao->addAttrib(caps,index,componentsPerVertex3D,GL_FLOAT,stride,offset,normalized,divisor);
  }

  nofTriangles = adj->getNofTriangles();
}



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

VSSV::VSSV(vars::Vars&vars):
  ShadowVolumes(vars       )
{
  assert(this!=nullptr);

  //compute adjacency of the model
  auto const adj = createAdjacency(vars);
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

#include"VSSVShaders.h"
#include"SilhouetteShaders.h"

  drawSidesProgram = std::make_shared<ge::gl::Program>(
      std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        vars.getBool("vssv.usePlanes"             )?ge::gl::Shader::define("USE_PLANES"               ):"",
        vars.getBool("vssv.useStrips"             )?ge::gl::Shader::define("USE_TRIANGLE_STRIPS"      ):"",
        vars.getBool("vssv.useAllOppositeVertices")?ge::gl::Shader::define("USE_ALL_OPPOSITE_VERTICES"):"",
        silhouetteFunctions,
        _drawSidesVertexShaderSrc));

  createCapDataUsingPoints(adj);

  drawCapsProgram = std::make_shared<ge::gl::Program>(
      std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER,
        "#version 450\n",
        silhouetteFunctions,
        _drawCapsVertexShaderSrc));
}

VSSV::~VSSV(){}

void VSSV::drawSides(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
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
}

void VSSV::drawCaps(
    glm::vec4 const&lightPosition,
    glm::mat4 const&viewMatrix         ,
    glm::mat4 const&projectionMatrix   ){
  assert(this!=nullptr);
  assert(drawCapsProgram!=nullptr);
  assert(capsVao!=nullptr);
  drawCapsProgram->use();
  drawCapsProgram->setMatrix4fv("viewMatrix"      ,glm::value_ptr(viewMatrix      ));
  drawCapsProgram->setMatrix4fv("projectionMatrix",glm::value_ptr(projectionMatrix));
  drawCapsProgram->set4fv      ("lightPosition"   ,glm::value_ptr(lightPosition   ));
  capsVao->bind();
  size_t  const nofCapsPerTriangle = 2;
  GLuint  const nofInstances = GLuint(nofCapsPerTriangle * nofTriangles);
  GLsizei const nofVertices  = GLsizei(verticesPerTriangle);
  GLint   const firstVertex  = 0;
  glDrawArraysInstanced(GL_TRIANGLES,firstVertex,nofVertices,nofInstances);
  capsVao->unbind();
}

