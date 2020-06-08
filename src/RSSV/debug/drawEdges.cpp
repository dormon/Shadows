#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <imguiVars/addVarsLimits.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <FastAdjacency.h>

#include <RSSV/loadEdgeShader.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawEdges(vars::Vars&vars){
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
    fColor = vec4(0,1,0,1);
  }
  ).";

  std::string const gsSrc = R".(

layout(binding=0)buffer EdgePlanes{float edgePlanes[];};

layout(points)in;
layout(line_strip,max_vertices=2)out;

flat in uint vId[];

uniform mat4 view;
uniform mat4 proj;

void main(){
  uint edge = vId[0];

  vec3 P[2];
  loadEdge(P[0],P[1],edge);

  gl_Position = proj*view*vec4(P[0],1);EmitVertex();
  gl_Position = proj*view*vec4(P[1],1);EmitVertex();
  EndPrimitive();
}

  ).";


  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      Shader::define("ALIGN_SIZE",(uint32_t)alignSize),
      Shader::define("NOF_EDGES" ,(uint32_t)nofEdges ),
      loadEdgeShaderFWD,
      gsSrc,
      loadEdgeShader
      );
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc);

  vars.reCreate<Program>(
      "rssv.method.debug.drawEdgesProgram",
      vs,
      gs,
      fs
      );

}

void drawEdges(vars::Vars&vars){
  prepareDrawEdges(vars);

  auto const view          = *vars.get<glm::mat4>  ("rssv.method.debug.viewMatrix"        );
  auto const proj          = *vars.get<glm::mat4>  ("rssv.method.debug.projectionMatrix"  );
  auto const adj           =  vars.get<Adjacency>  ("adjacency"                           );
  auto const vao           =  vars.get<VertexArray>("rssv.method.debug.vao"               );
  auto const prg           =  vars.get<Program>    ("rssv.method.debug.drawEdgesProgram"  );
  auto const edges         =  vars.get<Buffer>     ("rssv.method.edgePlanes"              );

  auto nofEdges = adj->getNofEdges();

  edges->bindBase(GL_SHADER_STORAGE_BUFFER,0);

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
