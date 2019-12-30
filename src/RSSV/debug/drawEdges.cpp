#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <imguiVars/addVarsLimits.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>
#include <requiredBits.h>
#include <FastAdjacency.h>

#include <RSSV/debug/drawNodePool.h>

#include <RSSV/mortonShader.h>
#include <RSSV/quantizeZShader.h>
#include <RSSV/depthToZShader.h>
#include <RSSV/configShader.h>
#include <RSSV/config.h>


using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawEdges(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug"
      ,"wavefrontSize"                        
      );

  auto const alignedNofEdges =  vars.getUint32  ("rssv.method.alignedNofEdges"  );

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

layout(binding=0)buffer EdgeBuffer{float edgeBuffer[];};

layout(points)in;
layout(line_strip,max_vertices=4)out;

flat in uint vId[];

uniform mat4 view;
uniform mat4 proj;

void main(){
  uint edge = vId[0];

  vec4 P[2];
  P[0][0] = edgeBuffer[edge+0*ALIGNED_NOF_EDGES];
  P[0][1] = edgeBuffer[edge+1*ALIGNED_NOF_EDGES];
  P[0][2] = edgeBuffer[edge+2*ALIGNED_NOF_EDGES];
  P[0][3] = 1;
  P[1][0] = edgeBuffer[edge+3*ALIGNED_NOF_EDGES];
  P[1][1] = edgeBuffer[edge+4*ALIGNED_NOF_EDGES];
  P[1][2] = edgeBuffer[edge+5*ALIGNED_NOF_EDGES];
  P[1][3] = 1;

  gl_Position = proj*view*P[0];EmitVertex();
  gl_Position = proj*view*P[1];EmitVertex();
  EndPrimitive();

  //gl_Position = vec4(0,0,-1,1);EmitVertex();
  //gl_Position = vec4(1,1,-1,1);EmitVertex();
  //EndPrimitive();
}

  ).";


  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      Shader::define("ALIGNED_NOF_EDGES",alignedNofEdges),
      gsSrc);
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
  auto const edges         =  vars.get<Buffer>     ("rssv.method.edgeBuffer"              );

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
