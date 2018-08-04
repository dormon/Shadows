#include<CSSV/DrawCaps.h>
#include<CSSV/DrawCapsProgram.h>
#include<ShadowVolumes.h>
#include<geGL/StaticCalls.h>

using namespace std;
using namespace ge::gl;
using namespace glm;
using namespace cssv;

struct Vertex3D{
  float data[3];
};

struct Vertex4D{
  float data[4];
};

struct Triangle3D{
  Vertex3D data[3];
};

struct Triangle4D{
  Vertex4D data[3];
};

shared_ptr<VertexArray>createCapsVao(shared_ptr<Buffer>const&caps){
  auto vao = make_shared<VertexArray>();
  vao->addAttrib(caps,0,4,GL_FLOAT);
  return vao;
}

void copyVertex(Vertex4D & dst,Vertex3D const& src){
  memcpy(dst.data,src.data,sizeof(Vertex3D));
  dst.data[3] = 1.f;
}

void copyTriangle(Triangle4D& dst,Triangle3D const& src){
  for(size_t p=0;p<verticesPerTriangle;++p)
    copyVertex(dst.data[p],src.data[p]);
}

void copyTriangles(Triangle4D *const dst,Triangle3D const*const src,size_t nofTriangles){
  for(size_t t=0;t<nofTriangles;++t)
    copyTriangle(dst[t],src[t]);
}

shared_ptr<Buffer>createCapsBuffer(shared_ptr<Adjacency const>const&adj){
  auto const nofTriangles = adj->getNofTriangles();
  vector<Triangle4D>dst(nofTriangles);
  auto const dstPtr = dst.data();
  auto const srcPtr = reinterpret_cast<Triangle3D const*>(adj->getVertices().data());
  copyTriangles(dstPtr,srcPtr,nofTriangles);
  return make_shared<Buffer>(dst);
}

DrawCaps::DrawCaps(shared_ptr<Adjacency const>const&adj){
  caps         = createCapsBuffer(adj );
  vao          = createCapsVao   (caps);
  program      = createDrawCapsProgram();
  nofTriangles = adj->getNofTriangles();
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

