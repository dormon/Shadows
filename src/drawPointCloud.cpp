#include <Vars/Vars.h>
#include <FunctionPrologue.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <getMVP.h>
#include <addVarsLimits.h>
#include <util.h>
#include <RSSV/Hierarchy.h>
#include <Deferred.h>

using namespace glm;
using namespace ge::gl;
using namespace std;

void createPointCloudProgram(vars::Vars&vars){
  FUNCTION_PROLOGUE("all");
  auto const vsSrc = R".(
  #version 450
  #line 20

  layout(binding=0)uniform  sampler2D     positionTexture;
  layout(binding=1)uniform usampler2D        colorTexture;
  layout(binding=2)uniform  sampler2DRect    depthTexture;

  uniform mat4  pointCloudMVP = mat4(1);
  uniform mat4  mvp           = mat4(1);
  uniform float near          = 1.f;
  uniform float far           = 1000.f;
  uniform uint  selPoint      = 0;
  uniform uint  selLevel      = 0;
  uniform  int  drawColors    = 0;
  uniform uvec2 roundedSize   = uvec2(512,512);
  uniform  int  useDepth      = 0;
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

  uniform uvec2 tileGlobalSize[NOF_LEVELS] = {
    uvec2(64,64),
    uvec2(8 ,8 ),
    uvec2(1 ,1 ),
  };

  uniform uvec2 fullTileSize[NOF_LEVELS] = {
    uvec2(8,8),
    uvec2(8,8),
    uvec2(8,8),
  };

  uniform uvec2 fullTileMask[NOF_LEVELS] = {
    uvec2(7,7),
    uvec2(7,7),
    uvec2(7,7),
  };

  uniform uvec2 fullTileSizeAccumulated[NOF_LEVELS] = {
    uvec2(512,512),
    uvec2(64 ,64 ),
    uvec2(8  ,8  ),
  };

  uniform uvec2 accumulatedTileExponent[NOF_LEVELS] = {
    uvec2(6,6),
    uvec2(3,3),
    uvec2(0,0),
  };

  uniform uvec2 fullTileExponent[NOF_LEVELS] = {
    uvec2(3,3),
    uvec2(3,3),
    uvec2(3,3),
  };

  uniform uvec2 tileCound[NOF_LEVELS] = {
    uvec2(8 ,8 ),
    uvec2(64,64),
    uvec2(8 ,8 ),
  };

  /*
  void computeThreadCoords(out uvec2 localInvocationID, out uvec2 workGroupID, in uint threadID,uvec2 levelSize,uvec2 tileSize){
    uvec2 globalInvocationID = uvec2(threadID % levelSize.x , threadID % levelSize.x);
    localInvocationID = globalInvocationID % tileSize;
    workGroupID = globalInvocationID / tileSize;
  }

  #define SECOND_LAST_LEVEL (NOF_LEVELS-2)
  #define LAST_LEVEL        (NOF_LEVELS-1)

  uvec2 computeGlobalTileCoord(uint level,uvec2 globalInvocationID){
    return globalInvocationID >> accumulatedTileExponent[level];
  }

  uvec2 computeLocalCoord(uint level,uvec2 globalInvocationID){
    return computeGlobalTileCoord(level,globalInvocationID) & fullTileMask[level];
  }

  uvec2 threadIdGlobalInvocationID(uint id,uint levelWidth){
    return uvec2(id % levelWidth , id / levelWidth);
  }

  uvec2 toSnake(uvec2 coord,uvec2 size,uint yAxis){
    uint odd = coord[yAxis];
    uvec2 result;
    result[1-yAxis] = coord[1-yAxis] + odd*uint(size[1-yAxis] - 2*coord[1-yAxis] - 1u);
    result[  yAxis] = coord[  yAxis]                                                  ;
    return result;
  }

  #define O_BOTTOM      0
  #define O_LEFT        2
  #define O_LEFT_LAST   3
  #define O_TOP         4
  #define O_TOP_LAST    5

  #define O_LEFT        5
  #define O_BOTTOM      6
  #define O_BOTTOM_LAST 7
  #define O_RIGHT       8
  #define O_RIGHT_LAST  9

  uint snakeToOrientation(uvec2 coord,uvec2 count,uint yAxis){
    //odd,last,first
    if(!(coord[yAxis]&1) && (coord[1-yAxis] < count-1))return O_BOTTOM;
  }
  

  uvec2 idToSnake(uint id,uint levelWidth,uint startO){
    uvec2 threadCoord = threadIdToCoord(id,levelWidth);
    uvec2 l0 = (threadCoord>>tileGlobalExponent[0])&tileGlobalMask[0];
    uvec2 l1 = (threadCoord>>tileGlobalExponent[1])&tileGlobalMask[1];
    uvec2 l2 = (threadCoord>>tileGlobalExponent[2])&tileGlobalMask[2];
    uvec2 s0 = toSnake(l0,tileGlobalSize[0],1);
  }

  */

  /*
  uvec2 toCoord(uint id){
    uint l0 = (id / (1*64*64)) % 64;
    uint l1 = (id / (1*64   )) % 64;
    uint l2 = (id / (1      )) % 64;

    uvec2 c0 = uvec2(l0 % fullTileSize[0] , l0 / fullTileSize[0].x);
    uvec2 c1 = uvec2(l1 % fullTileSize[1] , l1 / fullTileSize[1].x);
    uvec2 c2 = uvec2(l2 % fullTileSize[2] , l2 / fullTileSize[2].x);
  }
  */

  float depthToZ(float d){
    return 2*near*far/(d*(far-near)-far-near);
  }

  vec4 getWorldSpacePosition(uvec2 coord){
    if(useDepth == 0)
      return texelFetch(positionTexture,ivec2(coord),0);

    uvec2 size = textureSize(depthTexture);
    float depth = texelFetch(depthTexture,ivec2(coord)).x*2.f-1.f;
    float z = depthToZ(depth);
    
    return inverse(pointCloudMVP)*vec4((2*vec2(coord) / vec2(size) - 1) * (z),depth*(z),z);
  }

  void main(){
    uvec2 size = textureSize(colorTexture,0);

    uvec2 coord    = ivec2(gl_VertexID%size.x,gl_VertexID/size.x);

    uvec2 globalInvocationID = uvec2(gl_VertexID % roundedSize.x , gl_VertexID / roundedSize.x);

    if(any(greaterThanEqual(globalInvocationID,size))){
      gl_Position = vec4(2,0,0,1);
      return;
    }

    vec4 position = getWorldSpacePosition(coord);


    if(drawColors == 1){
      uvec4 color = texelFetch(colorTexture   ,ivec2(coord),0);
      vColor      = vec3((color.xyz>>0u)&0xffu)/0xffu;
    }else{
      //uvec2 tileCoord    = threadIdToTileCoord(gl_VertexID,size.x,tileGlobalExponent[selLevel],tileGlobalMask[selLevel]);
      vColor = vec3(1);//(tileCoord.x&1u) ^ (tileCoord.y&1u));
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

void computeHierarchy(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","wavefrontSize","pointCloud.positionTexture");

  std::cerr << "computeHierarchy" << std::endl;
  auto tex = vars.get<Texture>("pointCloud.positionTexture");
  auto size = uvec2(tex->getWidth(0),tex->getHeight(0));
  auto const wavefrontSize = vars.getSizeT("wavefrontSize");
  auto hier = vars.reCreate<rssv::Hierarchy>("pointCloud.hierarchy",size,wavefrontSize);

  vars.reCreate<uint32_t>("pointCloud.hierarchy.nofLevels",hier->nofLevels);
}

void drawPointCloud(vars::Vars&vars){
  FUNCTION_CALLER();
  if(!vars.has("pointCloud.positionTexture"))return;

  createPointCloudProgram(vars);
  createPointCloudVAO(vars);
  computeHierarchy(vars);

  glClearColor(0,.1,0,1);
  glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

  auto window          = vars.get<uvec2      >("windowSize"                );
  auto program         = vars.get<Program    >("pointCloud.program"        );
  auto vao             = vars.get<VertexArray>("pointCloud.vao"            );
  auto pointCloud      = vars.get<Texture    >("pointCloud.positionTexture");
  auto pointCloudColor = vars.get<Texture    >("pointCloud.colorTexture"   );
  auto pointCloudDepth = vars.get<Texture    >("pointCloud.depthTexture"   );
  auto pointCloudMVP   = vars.get<mat4       >("pointCloud.mvp"            );
  auto near            = vars.getFloat        ("pointCloud.near"           );
  auto far             = vars.getFloat        ("pointCloud.far"            );
  
  uint32_t selPointX = vars.addOrGetUint32("pointCloud.selPointX");
  uint32_t selPointY = vars.addOrGetUint32("pointCloud.selPointY");
  uint32_t selPoint  = vars.addOrGetUint32("pointCloud.selPoint" );
  uint32_t selLevel  = vars.addOrGetUint32("pointCloud.selLevel",2);
  bool     useDepth  = vars.addOrGetBool  ("pointCloud.useDepth");
  bool     usePointID     = vars.addOrGetBool  ("pointCloud.usePointID"     );
  bool     drawColors     = vars.addOrGetBool  ("pointCloud.drawColors");

  uint32_t width  = pointCloud->getWidth(0);
  uint32_t height = pointCloud->getHeight(0);
  addVarsLimitsU(vars,"pointCloud.selPointX",0,width-1,1);
  addVarsLimitsU(vars,"pointCloud.selPointY",0,height-1,1);
  addVarsLimitsU(vars,"pointCloud.selPoint" ,0,height*width-1,1);
  addVarsLimitsU(vars,"pointCloud.selLevel" ,0,2,1);

  uint32_t ww = divRoundUp(width ,8)*8;
  uint32_t hh = divRoundUp(height,8)*8;

  auto mvp = getMVP(vars);
  vao->bind();
  pointCloud->bind(0);
  pointCloudColor->bind(1);
  //pointCloudDepth->bind(2);
  vars.get<GBuffer>("gBuffer")->depth->bind(2);
  program
    ->setMatrix4fv("mvp"          ,value_ptr(mvp           ))
    ->setMatrix4fv("pointCloudMVP",value_ptr(*pointCloudMVP))
    ->set1ui      ("selLevel"     ,selLevel                 )
    ->set1i       ("drawColors"   ,drawColors               )
    ->set1i       ("useDepth"     ,useDepth                 )
    ->set2ui      ("roundedSize"  ,ww,hh                    )
    ->set1f       ("near"         ,near                     )
    ->set1f       ("far"          ,far                      )
    ->use();
  

  if(usePointID)
    program->set1ui      ("selPoint" ,selPoint );
  else
    program->set1ui      ("selPoint" ,selPointY * width + selPointX );

  glDrawArrays(GL_POINTS,0,ww*hh);
  vao->unbind();
}

// threadID.xy workGroupID.xy
//
// tileSize.x = fullTileSize.x | borderTileSize.x
// tileSize.y = fullTileSize.y | borderTileSize.y
// BOTTOM
// LEFT
// TOP
// RIGHT
//
// L -> B*LLT*
// B -> L*BBR*
// R -> T*RRB*
// T -> R*TTL*
//
//
