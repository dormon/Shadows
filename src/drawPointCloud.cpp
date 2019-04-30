#include <Vars/Vars.h>
#include <FunctionPrologue.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <getMVP.h>
#include <addVarsLimits.h>

using namespace glm;
using namespace ge::gl;
using namespace std;

void createPointCloudProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("all");
  auto const vsSrc = R".(
  #version 450
  layout(binding=0)uniform  sampler2D positionTexture;
  layout(binding=1)uniform usampler2D    colorTexture;
  uniform mat4 mvp;
  uniform uint selectedPoint = 0;
  out vec3 vColor;
  void main(){
    uvec2 size     = textureSize(positionTexture,0);
    ivec2 coord    = ivec2(gl_VertexID%size.x,gl_VertexID/size.x);
    vec4  position = texelFetch(positionTexture,coord,0);
    uvec4 color    = texelFetch(colorTexture   ,coord,0);
    vColor         = vec3((color.xyz>>0u)&0xffu)/0xffu;

    if(gl_VertexID == selectedPoint){
      gl_PointSize = 10;
      vColor = vec3(1,0,0);
    }else
      gl_PointSize = 1;
    gl_Position = mvp * position;
  }
  ).";
  auto const vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);

  auto const fsSrc = R".(
  #version 450
  layout(location=0)out vec4 fColor;
  in vec3 vColor;
  void main(){
    fColor = vec4(vColor,1);
  }
  ).";
  auto const fs = make_shared<Shader>(GL_FRAGMENT_SHADER,fsSrc);
  vars.add<Program>("drawPointCloudProgram",vs,fs);
}

void createPointCloudVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("all");
  vars.add<VertexArray>("pointCloudVAO");
}

void drawPointCloud(vars::Vars&vars){
  FUNCTION_CALLER();
  if(!vars.has("gBufferAsPointCloud"))return;

  uint32_t selectedPoint = vars.addOrGetUint32("selectedPoint");
  addVarsLimitsU(vars,"selectedPoint",0,100,10);
  createPointCloudProgram(vars);
  createPointCloudVAO(vars);

  glClearColor(0,.1,0,1);
  glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

  auto program         = vars.get<Program    >("drawPointCloudProgram"   );
  auto vao             = vars.get<VertexArray>("pointCloudVAO"           );
  auto window          = vars.get<uvec2      >("windowSize"              );
  auto pointCloud      = vars.get<Texture    >("gBufferAsPointCloud"     );
  auto pointCloudColor = vars.get<Texture    >("gBufferAsPointCloudColor");
  
  auto mvp = getMVP(vars);
  vao->bind();
  pointCloud->bind(0);
  pointCloudColor->bind(1);
  program->setMatrix4fv("mvp",value_ptr(mvp));
  program->set1ui("selectedPoint",selectedPoint);
  program->use();
  glDrawArrays(GL_POINTS,0,window->x*window->y);
  vao->unbind();
}
