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
  uniform uint selPoint   = 0;
  uniform uint selLevel   = 0;
  uniform  int drawColors = 0;
  out vec3 vColor;

  #ifndef NOF_LEVELS
  #define NOF_LEVELS 3
  #endif//NOF_LEVELS


  uniform uvec2 tileGlobalExponent[NOF_LEVELS] = {
    uvec2(6),
    uvec2(3),
    uvec2(0),
  };

  uniform uvec2 tileGlobalMask[NOF_LEVELS] = {
    uvec2(7),
    uvec2(7),
    uvec2(7),
  };

  uvec2 threadIdToCoord(uint id,uint levelWidth){
    return uvec2(id % levelWidth , id / levelWidth);
  }

  uvec2 threadIdToTileCoord(uint id,uint levelWidth,uvec2 tileExponent,uvec2 tileMask){
    uvec2 threadCoord = threadIdToCoord(id,levelWidth);
    return (threadCoord>>tileExponent)&tileMask;
  }

  uvec2 toSnake(uvec2 coord,uvec2 size,uint yAxis){
    uint odd = coord[yAxis];
    uvec2 result;
    result[1-yAxis] = coord[1-yAxis] + odd*uint(size[1-yAxis] - 2*coord[1-yAxis] - 1u);
    result[  yAxis] = coord[  yAxis]                                                  ;
    return result;
  }


  void main(){
    uvec2 size     = textureSize(positionTexture,0);
    uvec2 coord    = ivec2(threadIdToCoord(gl_VertexID,size.x));


    vec4  position = texelFetch(positionTexture,ivec2(coord),0);

    if(drawColors == 1){
      uvec4 color = texelFetch(colorTexture   ,ivec2(coord),0);
      vColor      = vec3((color.xyz>>0u)&0xffu)/0xffu;
    }else{
      uvec2 tileCoord    = threadIdToTileCoord(gl_VertexID,size.x,tileGlobalExponent[selLevel],tileGlobalMask[selLevel]);
      vColor = vec3((tileCoord.x&1u) ^ (tileCoord.y&1u));
    }

    if(gl_VertexID == selPoint){
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
  vars.add<Program>("pointCloud.program",vs,fs);
}

void createPointCloudVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("all");
  vars.add<VertexArray>("pointCloud.vao");
}

void drawPointCloud(vars::Vars&vars){
  FUNCTION_CALLER();
  if(!vars.has("pointCloud.positionTexture"))return;

  createPointCloudProgram(vars);
  createPointCloudVAO(vars);

  glClearColor(0,.1,0,1);
  glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

  auto window          = vars.get<uvec2      >("windowSize"                );
  auto program         = vars.get<Program    >("pointCloud.program"        );
  auto vao             = vars.get<VertexArray>("pointCloud.vao"            );
  auto pointCloud      = vars.get<Texture    >("pointCloud.positionTexture");
  auto pointCloudColor = vars.get<Texture    >("pointCloud.colorTexture"   );
  
  uint32_t selPointX = vars.addOrGetUint32("pointCloud.selPointX");
  uint32_t selPointY = vars.addOrGetUint32("pointCloud.selPointY");
  uint32_t selPoint  = vars.addOrGetUint32("pointCloud.selPoint" );
  uint32_t selLevel  = vars.addOrGetUint32("pointCloud.selLevel",2);
  bool     usePointID     = vars.addOrGetBool  ("pointCloud.usePointID"     );
  bool     drawColors     = vars.addOrGetBool  ("pointCloud.drawColors");

  uint32_t width = pointCloud->getWidth(0);
  uint32_t height = pointCloud->getHeight(0);
  addVarsLimitsU(vars,"pointCloud.selPointX",0,width-1,1);
  addVarsLimitsU(vars,"pointCloud.selPointY",0,height-1,1);
  addVarsLimitsU(vars,"pointCloud.selPoint" ,0,height*width-1,1);
  addVarsLimitsU(vars,"pointCloud.selLevel" ,0,2,1);

  auto mvp = getMVP(vars);
  vao->bind();
  pointCloud->bind(0);
  pointCloudColor->bind(1);
  program
    ->setMatrix4fv("mvp"       ,value_ptr(mvp))
    ->set1ui      ("selLevel"  ,selLevel      )
    ->set1i       ("drawColors",drawColors    )
    ->use();
  
  if(usePointID)
    program->set1ui      ("selPoint" ,selPoint );
  else
    program->set1ui      ("selPoint" ,selPointY * width + selPointX );

  glDrawArrays(GL_POINTS,0,width*height);
  vao->unbind();
}
