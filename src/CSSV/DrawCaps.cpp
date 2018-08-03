#include<CSSV/DrawCaps.h>
#include<CSSV/DrawCapsProgram.h>
#include<ShadowVolumes.h>
#include<geGL/StaticCalls.h>

using namespace std;
using namespace ge::gl;
using namespace glm;
using namespace cssv;

#define ___ std::cerr << __FILE__ << " " << __LINE__ << std::endl

shared_ptr<VertexArray>createCapsVao(shared_ptr<Buffer>const&caps){
  ___;
  auto vao = make_shared<VertexArray>();
  ___;
  vao->addAttrib(caps,0,4,GL_FLOAT);
  ___;
  return vao;
}

shared_ptr<Buffer>createCapsBuffer(shared_ptr<Adjacency const>const&adj){
  ___;
  auto const nofTriangles = adj->getNofTriangles();
  vector<float>dst(componentsPerVertex4D*verticesPerTriangle*nofTriangles);

  auto const dstPtr = dst.data();
  auto const srcPtr = adj->getVertices().data();
  ___;
  for(size_t t=0;t<nofTriangles;++t){
    std::cerr << __FILE__ << " triangle: " << t << std::endl;
    auto const triangleDstPtr = dstPtr + t*componentsPerVertex4D*verticesPerTriangle;
    auto const triangleSrcPtr = srcPtr + t*componentsPerVertex3D*verticesPerTriangle;
    for(size_t p=0;p<verticesPerTriangle;++p){
      auto   const vertexDstPtr = triangleDstPtr + p*componentsPerVertex4D;
      auto   const vertexSrcPtr = triangleSrcPtr + p*componentsPerVertex3D;
      size_t const sizeofVertex3DInBytes = componentsPerVertex3D * sizeof(float);
      memcpy(vertexDstPtr,vertexSrcPtr,sizeofVertex3DInBytes);
      vertexDstPtr[3] = 1.f;
    }
  }
  ___;
  return make_shared<Buffer>(dst);
}

DrawCaps::DrawCaps(shared_ptr<Adjacency const>const&adj){
  ___;
  caps    = createCapsBuffer(adj );
  ___;
  vao     = createCapsVao   (caps);
  ___;
  program = createDrawCapsProgram();
  ___;
  nofTriangles = adj->getNofTriangles();
  ___;
}

void DrawCaps::draw(
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix){
  auto mvp = projectionMatrix * viewMatrix;
  program
    ->setMatrix4fv("mvp"          ,value_ptr(mvp          ))
    ->set4fv      ("lightPosition",value_ptr(lightPosition));
  program->use();
  vao->bind();
  glDrawArrays(GL_TRIANGLES,0,(GLsizei)nofTriangles*3);
  vao->unbind();
}

