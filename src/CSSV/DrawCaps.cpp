#include<CSSV/DrawCaps.h>
#include<CSSV/DrawCapsProgram.h>
#include<ShadowVolumes.h>
#include<geGL/StaticCalls.h>
#include<Simplex.h>

using namespace std;
using namespace ge::gl;
using namespace glm;
using namespace cssv;

shared_ptr<VertexArray>createCapsVao(shared_ptr<Buffer>const&caps){
  auto vao = make_shared<VertexArray>();
  vao->addAttrib(caps,0,4,GL_FLOAT);
  return vao;
}

void copyTriangles(Triangle4Df *const dst,Triangle3Df const*const src,size_t nofTriangles){
  for(size_t t=0;t<nofTriangles;++t)
    createHomogenous(dst[t],src[t]);
}

shared_ptr<Buffer>createCapsBuffer(Adjacency const*adj){
  auto const nofTriangles = adj->getNofTriangles();
  vector<Triangle4Df>dst(nofTriangles);
  auto const dstPtr = dst.data();
  auto const srcPtr = reinterpret_cast<Triangle3Df const*>(adj->getVertices().data());
  copyTriangles(dstPtr,srcPtr,nofTriangles);
  return make_shared<Buffer>(dst);
}

DrawCaps::DrawCaps(Adjacency const*adj){
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

