#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <imguiVars/addVarsLimits.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <FastAdjacency.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawSilhouettes(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug"
      ,"wavefrontSize"                        
      ,"rssv.param.alignment"
      ,"adjacency"
      );

  auto const alignSize                    =  vars.getSizeT       ("rssv.param.alignment"                   );
  auto adj                                =  vars.get<Adjacency> ("adjacency"                              );
  auto const nofEdges                     =  adj->getNofEdges();

  std::string const vsSrc = R".(
  #version 450

  flat out uint vId;
  void main(){
    vId = gl_VertexID;
  }

  ).";

  std::string const fsSrc = R".(

  layout(location=0)out vec4 fColor;
  void main(){
    fColor = vec4(1,0,0,1);
  }
  ).";

  std::string const gsSrc = R".(

layout(binding=0)readonly buffer EdgePlanes       {float edgePlanes       [];};
layout(binding=1)readonly buffer MultBuffer       {
  uint  nofSilhouettes  ;
  uint  multBuffer    [];
};


layout(points)in;
layout(line_strip,max_vertices=2)out;

flat in uint vId[];

uniform mat4 view;
uniform mat4 proj;

#ifndef ALIGN_SIZE
#define ALIGN_SIZE 128
#endif//ALIGN_SIZE
#define ALIGN(W,A) uint(uint(uint(W)/uint(A))*uint(A) + uint((uint(W)%uint(A))!=0u)*uint(A))
#define ALIGN_SIZE_FLOAT ALIGN(ALIGN_SIZE,4u)
#define ALIGN_OFFSET(i) uint(ALIGN(NOF_EDGES,ALIGN_SIZE_FLOAT)*uint(i))

void main(){
  uint thread = vId[0];

  if(thread >= nofSilhouettes)return;

  uint res = multBuffer[thread];
  uint edge = res & 0x1fffffffu;
  int  mult = int(res) >> 29;

  vec4 P[2];
  P[0][0] = edgePlanes[edge+ALIGN_OFFSET(0)];
  P[0][1] = edgePlanes[edge+ALIGN_OFFSET(1)];
  P[0][2] = edgePlanes[edge+ALIGN_OFFSET(2)];
  P[0][3] = 1;
  P[1][0] = edgePlanes[edge+ALIGN_OFFSET(3)];
  P[1][1] = edgePlanes[edge+ALIGN_OFFSET(4)];
  P[1][2] = edgePlanes[edge+ALIGN_OFFSET(5)];
  P[1][3] = 1;

  gl_Position = proj*view*P[0];EmitVertex();
  gl_Position = proj*view*P[1];EmitVertex();
  EndPrimitive();
}

  ).";


  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      Shader::define("ALIGN_SIZE",(uint32_t)alignSize),
      Shader::define("NOF_EDGES" ,(uint32_t)nofEdges ),
      gsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc);

  vars.reCreate<Program>(
      "rssv.method.debug.drawSilhouettesProgram",
      vs,
      gs,
      fs
      );

}

void drawSilhouettes(vars::Vars&vars){
  prepareDrawSilhouettes(vars);
  auto const view              = *vars.get<glm::mat4>  ("rssv.method.debug.viewMatrix"            );
  auto const proj              = *vars.get<glm::mat4>  ("rssv.method.debug.projectionMatrix"      );
  auto const adj               =  vars.get<Adjacency>  ("adjacency"                               );
  auto const vao               =  vars.get<VertexArray>("rssv.method.debug.vao"                   );
  auto const prg               =  vars.get<Program>    ("rssv.method.debug.drawSilhouettesProgram");
  auto const edgePlanes        =  vars.get<Buffer>     ("rssv.method.edgePlanes"                  );
  auto const multBuffer        =  vars.get<Buffer>     ("rssv.method.debug.dump.multBuffer"       );
  auto nofEdges = adj->getNofEdges();

  edgePlanes->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  multBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,1);

  vao->bind();
  prg->use();
  prg
    ->setMatrix4fv("view"         ,glm::value_ptr(view    ))
    ->setMatrix4fv("proj"         ,glm::value_ptr(proj    ))
    ;

  glDrawArrays(GL_POINTS,0,nofEdges);

  vao->unbind();

}

}
