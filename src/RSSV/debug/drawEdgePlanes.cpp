#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <imguiVars/addVarsLimits.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

#include <FunctionPrologue.h>
#include <FastAdjacency.h>

#include <RSSV/getEdgePlanesShader.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareDrawEdgePlanes(vars::Vars&vars){
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
    fColor = vec4(0,0,0,1);
  }
  ).";

  std::string const gsSrc = R".(

layout(binding=0)buffer EdgeBuffer       {float edgeBuffer       [];};
layout(binding=1)buffer MultBuffer       {
  uint  nofSilhouettes  ;
  uint  multBuffer    [];
};

layout(points)in;
layout(line_strip,max_vertices=20)out;

flat in uint vId[];

uniform mat4 view;
uniform mat4 proj;

uniform mat4 debugView;
uniform mat4 debugProj;

uniform vec4 light;
uniform int selectedEdge = -1;

void main(){
  uint thread = vId[0];

  if(thread >= nofSilhouettes)return;
  if(selectedEdge >= 0 && int(thread) != selectedEdge)return;

  uint res = multBuffer[thread];
  uint edge = res & 0x1fffffffu;
  int  mult = int(res) >> 29;

  vec4 P[2];
  P[0][0] = edgeBuffer[edge+0*ALIGNED_NOF_EDGES];
  P[0][1] = edgeBuffer[edge+1*ALIGNED_NOF_EDGES];
  P[0][2] = edgeBuffer[edge+2*ALIGNED_NOF_EDGES];
  P[0][3] = 1;
  P[1][0] = edgeBuffer[edge+3*ALIGNED_NOF_EDGES];
  P[1][1] = edgeBuffer[edge+4*ALIGNED_NOF_EDGES];
  P[1][2] = edgeBuffer[edge+5*ALIGNED_NOF_EDGES];
  P[1][3] = 1;

  if(debugProj[0][0]==1337 && debugView[0][0] == 1337)return;
  mat4 debugM = debugProj * debugView;
  vec4 aa = P[0];
  vec4 bb = P[1];
  vec4 cc = light;

  aa = debugM*aa;
  bb = debugM*bb;
  cc = debugM*cc;

  vec4 edgePlane;
  vec4 aPlane;
  vec4 bPlane;
  vec4 abPlane;

  getEdgePlanes(edgePlane,aPlane,bPlane,abPlane,aa,bb,cc);

  mat4 invTranDebug = transpose(inverse(debugM));

  edgePlane = invTranDebug * edgePlane;
  aPlane    = invTranDebug * aPlane   ;
  bPlane    = invTranDebug * bPlane   ;
  abPlane   = invTranDebug * abPlane  ;

  vec3 a  = P[0].xyz;
  vec3 b  = P[1].xyz;
  vec3 l  = light.xyz;

  float sc = 10;
  vec3 en = 0.1*normalize(edgePlane.xyz);
  vec3 al = sc*normalize(a-l);
  vec3 bl = sc*normalize(b-l);

  mat4 M = proj*view;


  gl_Position = M * vec4(a+al,1);EmitVertex();
  gl_Position = M * vec4(a   ,1);EmitVertex();
  gl_Position = M * vec4(b   ,1);EmitVertex();
  gl_Position = M * vec4(b+bl,1);EmitVertex();
  EndPrimitive();
  gl_Position = M * vec4(a+al+en,1);EmitVertex();
  gl_Position = M * vec4(a   +en,1);EmitVertex();
  gl_Position = M * vec4(a   -en,1);EmitVertex();
  gl_Position = M * vec4(a+al-en,1);EmitVertex();
  EndPrimitive();
  gl_Position = M * vec4(b+bl+en,1);EmitVertex();
  gl_Position = M * vec4(b   +en,1);EmitVertex();
  gl_Position = M * vec4(b   -en,1);EmitVertex();
  gl_Position = M * vec4(b+bl-en,1);EmitVertex();
  EndPrimitive();


  gl_Position = M * vec4(a+en,1);EmitVertex();
  gl_Position = M * vec4(b+en,1);EmitVertex();
  EndPrimitive();

  gl_Position = M * vec4(a-en,1);EmitVertex();
  gl_Position = M * vec4(b-en,1);EmitVertex();
  EndPrimitive();

}

  ).";


  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      Shader::define("ALIGNED_NOF_EDGES",alignedNofEdges),
      getEdgePlanesShader,
      gsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc);

  vars.reCreate<Program>(
      "rssv.method.debug.drawEdgePlanesProgram",
      vs,
      gs,
      fs
      );

}

void drawEdgePlanes(vars::Vars&vars){
  prepareDrawEdgePlanes(vars);
  auto const view              = *vars.get<glm::mat4>  ("rssv.method.debug.viewMatrix"            );
  auto const proj              = *vars.get<glm::mat4>  ("rssv.method.debug.projectionMatrix"      );

  auto const debugLight        = *vars.get<glm::vec4 >("rssv.method.debug.dump.lightPosition"     );
  auto const debugView         = *vars.get<glm::mat4 >("rssv.method.debug.dump.viewMatrix"        );
  auto const debugProj         = *vars.get<glm::mat4 >("rssv.method.debug.dump.projectionMatrix"  );


  auto const adj               =  vars.get<Adjacency>  ("adjacency"                               );
  auto const vao               =  vars.get<VertexArray>("rssv.method.debug.vao"                   );
  auto const prg               =  vars.get<Program>    ("rssv.method.debug.drawEdgePlanesProgram" );
  auto const edges             =  vars.get<Buffer>     ("rssv.method.edgeBuffer"                  );
  auto const multBuffer        =  vars.get<Buffer>     ("rssv.method.debug.dump.multBuffer"       );
  auto nofEdges = adj->getNofEdges();

  edges->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  multBuffer->bindBase(GL_SHADER_STORAGE_BUFFER,1);

  vao->bind();
  prg->use();
  prg
    ->setMatrix4fv("view"         ,glm::value_ptr(view      ))
    ->setMatrix4fv("proj"         ,glm::value_ptr(proj      ))
    ->setMatrix4fv("debugProj"    ,glm::value_ptr(debugProj ))
    ->setMatrix4fv("debugView"    ,glm::value_ptr(debugView ))
    ->set4fv      ("light"        ,glm::value_ptr(debugLight))
    ;

  prg->set1i("selectedEdge",vars.addOrGetInt32("rssv.param.selectedEdge",-1));

  glDrawArrays(GL_POINTS,0,nofEdges);

  vao->unbind();

}

}
