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

#include <Sintorn2/debug/drawNodePool.h>

#include <Sintorn2/mortonShader.h>
#include <Sintorn2/quantizeZShader.h>
#include <Sintorn2/depthToZShader.h>
#include <Sintorn2/configShader.h>
#include <Sintorn2/config.h>


using namespace ge::gl;
using namespace std;

namespace sintorn2::debug{

void prepareDrawSF(vars::Vars&vars){
  FUNCTION_PROLOGUE("sintorn2.method.debug"
      ,"wavefrontSize"                        
      ,"sintorn2.method.debug.dump.config"    
      ,"sintorn2.method.debug.dump.near"      
      ,"sintorn2.method.debug.dump.far"       
      ,"sintorn2.method.debug.dump.fovy"      
      ,"sintorn2.param.morePlanes"
      );

  auto const cfg            = *vars.get<Config>        ("sintorn2.method.debug.dump.config"    );


  std::string const vsSrc = R".(
  #version 450

  flat out uint vId;
  void main(){
    vId = gl_VertexID;
  }

  ).";

  std::string const fsSrc = R".(

  layout(location=0)out vec4 fColor;
  in vec3 gColor;
  void main(){
    fColor = vec4(gColor,1);
  }
  ).";

  std::string const gsSrc = R".(

#ifndef SF_ALIGNMENT
#define SF_ALIGNMENT 128
#endif//SF_ALIGNMENT

#ifndef SF_INTERLEAVE
#define SF_INTERLEAVE 0
#endif//SF_INTERLEAVE

#ifndef MORE_PLANES
#define MORE_PLANES 0
#endif//MORE_PLANES

const uint planesPerSF    = 4u + MORE_PLANES*3u;
const uint floatsPerPlane = 4u;
const uint floatsPerSF    = planesPerSF * floatsPerPlane;

const uint alignedNofSF = (uint(NOF_TRIANGLES / SF_ALIGNMENT) + uint((NOF_TRIANGLES % SF_ALIGNMENT) != 0u)) * SF_ALIGNMENT;

uint divRoundUp(uint x,uint y){
  return uint(x/y) + uint((x%y)>0);
}

layout(points)in;
layout(line_strip,max_vertices=28)out;

flat in uint vId[];

layout(std430,binding=0)buffer ShadowFrusta{float shadowFrusta[];};

uniform mat4 view;
uniform mat4 proj;

uniform mat4 nodeView;
uniform mat4 nodeProj;

uniform vec4 lightPosition;

out vec3 gColor;

void main(){
  uint gid = vId[0];

  vec4 e0;
  vec4 e1;
  vec4 e2;
  vec4 e3;

#if SF_INTERLEAVE == 1
  e0[0] = shadowFrusta[alignedNofSF* 0u + gid];
  e0[1] = shadowFrusta[alignedNofSF* 1u + gid];
  e0[2] = shadowFrusta[alignedNofSF* 2u + gid];
  e0[3] = shadowFrusta[alignedNofSF* 3u + gid];
                                              
  e1[0] = shadowFrusta[alignedNofSF* 4u + gid];
  e1[1] = shadowFrusta[alignedNofSF* 5u + gid];
  e1[2] = shadowFrusta[alignedNofSF* 6u + gid];
  e1[3] = shadowFrusta[alignedNofSF* 7u + gid];
                                              
  e2[0] = shadowFrusta[alignedNofSF* 8u + gid];
  e2[1] = shadowFrusta[alignedNofSF* 9u + gid];
  e2[2] = shadowFrusta[alignedNofSF*10u + gid];
  e2[3] = shadowFrusta[alignedNofSF*11u + gid];
                                              
  e3[0] = shadowFrusta[alignedNofSF*12u + gid];
  e3[1] = shadowFrusta[alignedNofSF*13u + gid];
  e3[2] = shadowFrusta[alignedNofSF*14u + gid];
  e3[3] = shadowFrusta[alignedNofSF*15u + gid];
#else
  e0[0] = shadowFrusta[gid*floatsPerSF+ 0u];
  e0[1] = shadowFrusta[gid*floatsPerSF+ 1u];
  e0[2] = shadowFrusta[gid*floatsPerSF+ 2u];
  e0[3] = shadowFrusta[gid*floatsPerSF+ 3u];
  e1[0] = shadowFrusta[gid*floatsPerSF+ 4u];
  e1[1] = shadowFrusta[gid*floatsPerSF+ 5u];
  e1[2] = shadowFrusta[gid*floatsPerSF+ 6u];
  e1[3] = shadowFrusta[gid*floatsPerSF+ 7u];
  e2[0] = shadowFrusta[gid*floatsPerSF+ 8u];
  e2[1] = shadowFrusta[gid*floatsPerSF+ 9u];
  e2[2] = shadowFrusta[gid*floatsPerSF+10u];
  e2[3] = shadowFrusta[gid*floatsPerSF+11u];
  e3[0] = shadowFrusta[gid*floatsPerSF+12u];
  e3[1] = shadowFrusta[gid*floatsPerSF+13u];
  e3[2] = shadowFrusta[gid*floatsPerSF+14u];
  e3[3] = shadowFrusta[gid*floatsPerSF+15u];
#endif


  //A = inverse(transpose(proj * view)) * B
  //A = M * B
  //inv(M) * A = inv(M) * M * B
  //inv(M) * A = B
  //


  mat4 back = transpose(nodeProj*nodeView);
  e0 = back*e0;
  e1 = back*e1;
  e2 = back*e2;
  e3 = back*e3;

  vec3 n0 = normalize(e0.xyz);
  vec3 n1 = normalize(e1.xyz);
  vec3 n2 = normalize(e2.xyz);
  vec3 n3 = normalize(e3.xyz);

  vec3 l0 = normalize(cross(e2.xyz,e0.xyz));
  vec3 l1 = normalize(cross(e0.xyz,e1.xyz));
  vec3 l2 = normalize(cross(e1.xyz,e2.xyz));

  //dot(e3.xyz,(lightPosition.xyz + t*l0)) + e3.w == 0
  //dot(e3.xyz,lightPosition.xyz) + t*dot(l0,e3.xyz + e3.w == 0
  //t = (-e3.w - dot(e3.xyz,lightPosition.xyz)) / dot(l0,e3.xyz))

  float t0 = (-e3.w - dot(e3.xyz,lightPosition.xyz)) / dot(l0,e3.xyz);
  float t1 = (-e3.w - dot(e3.xyz,lightPosition.xyz)) / dot(l1,e3.xyz);
  float t2 = (-e3.w - dot(e3.xyz,lightPosition.xyz)) / dot(l2,e3.xyz);


  vec3 v0 = lightPosition.xyz + t0*l0;
  vec3 v1 = lightPosition.xyz + t1*l1;
  vec3 v2 = lightPosition.xyz + t2*l2;

  vec3 v0l = normalize(v0 - lightPosition.xyz);
  vec3 v1l = normalize(v1 - lightPosition.xyz);
  vec3 v2l = normalize(v2 - lightPosition.xyz);

  vec3 w0 = v0 + v0l*30;
  vec3 w1 = v1 + v1l*30;
  vec3 w2 = v2 + v2l*30;


  mat4 M = proj*view;

  gColor = vec3(1,0,1);
  gl_Position = M*vec4(v0,1);EmitVertex();
  gl_Position = M*vec4(v1,1);EmitVertex();
  gl_Position = M*vec4(w1,1);EmitVertex();
  gl_Position = M*vec4(w0,1);EmitVertex();
  gl_Position = M*vec4(v0,1);EmitVertex();
  EndPrimitive();

  gl_Position = M*vec4(v1,1);EmitVertex();
  gl_Position = M*vec4(v2,1);EmitVertex();
  gl_Position = M*vec4(w2,1);EmitVertex();
  gl_Position = M*vec4(w1,1);EmitVertex();
  gl_Position = M*vec4(v1,1);EmitVertex();
  EndPrimitive();

  gl_Position = M*vec4(v2,1);EmitVertex();
  gl_Position = M*vec4(v0,1);EmitVertex();
  gl_Position = M*vec4(w0,1);EmitVertex();
  gl_Position = M*vec4(w2,1);EmitVertex();
  gl_Position = M*vec4(v2,1);EmitVertex();
  EndPrimitive();


  //n0 = normalize(cross(v1-v0,w0-v0));
  //n1 = normalize(cross(v2-v1,w1-v1));
  //n2 = normalize(cross(v0-v2,w2-v2));
  gColor = vec3(0,1,0);
  gl_Position = M*vec4((v0+v1+w0+w1)/4.f   ,1);EmitVertex();
  gl_Position = M*vec4((v0+v1+w0+w1)/4.f+n0,1);EmitVertex();
  EndPrimitive();

  gl_Position = M*vec4((v1+v2+w1+w2)/4.f   ,1);EmitVertex();
  gl_Position = M*vec4((v1+v2+w1+w2)/4.f+n1,1);EmitVertex();
  EndPrimitive();

  gl_Position = M*vec4((v0+v2+w0+w2)/4.f   ,1);EmitVertex();
  gl_Position = M*vec4((v0+v2+w0+w2)/4.f+n2,1);EmitVertex();
  EndPrimitive();

  gl_Position = M*vec4((v0+v1+v2)/3.f   ,1);EmitVertex();
  gl_Position = M*vec4((v0+v1+v2)/3.f+n3,1);EmitVertex();
  EndPrimitive();
}

  ).";

  auto const sfAlignment         = vars.getUint32("sintorn2.param.sfAlignment"       );
  auto const sfInterleave        = vars.getInt32 ("sintorn2.param.sfInterleave"      );
  auto const nofTriangles        = vars.getUint32("sintorn2.method.nofTriangles"      );
  auto const morePlanes          = vars.getInt32 ("sintorn2.param.morePlanes"        );

  auto vs = make_shared<Shader>(GL_VERTEX_SHADER,vsSrc);
  auto gs = make_shared<Shader>(GL_GEOMETRY_SHADER,
      "#version 450\n",
      Shader::define("SF_ALIGNMENT"       ,(uint32_t)sfAlignment       ),
      Shader::define("SF_INTERLEAVE"      ,(int)     sfInterleave      ),
      Shader::define("NOF_TRIANGLES"      ,(uint32_t)nofTriangles      ),
      Shader::define("MORE_PLANES"        ,(int)     morePlanes        ),
      gsSrc);
  auto fs = make_shared<Shader>(GL_FRAGMENT_SHADER,
      "#version 450\n",
      fsSrc);

  vars.reCreate<Program>(
      "sintorn2.method.debug.drawSFProgram",
      vs,
      gs,
      fs
      );

}

void drawSF(vars::Vars&vars){
  prepareDrawSF(vars);

  auto const nodeView       = *vars.get<glm::mat4>     ("sintorn2.method.debug.dump.viewMatrix"      );
  auto const nodeProj       = *vars.get<glm::mat4>     ("sintorn2.method.debug.dump.projectionMatrix");
  auto const sf             =  vars.get<Buffer>        ("sintorn2.method.debug.dump.shadowFrusta"    );

  auto const view           = *vars.get<glm::mat4>     ("sintorn2.method.debug.viewMatrix"           );
  auto const proj           = *vars.get<glm::mat4>     ("sintorn2.method.debug.projectionMatrix"     );
  auto const nofTriangles   = vars.getUint32           ("sintorn2.method.nofTriangles"               );
  auto const lightPosition  = *vars.get<glm::vec4 >    ("sintorn2.method.debug.dump.lightPosition"   );

  auto vao = vars.get<VertexArray>("sintorn2.method.debug.vao");

  auto prg = vars.get<Program>("sintorn2.method.debug.drawSFProgram");

  vao->bind();
  sf->bindBase(GL_SHADER_STORAGE_BUFFER,0);
  prg->use();
  prg
    ->setMatrix4fv("nodeView"     ,glm::value_ptr(nodeView))
    ->setMatrix4fv("nodeProj"     ,glm::value_ptr(nodeProj))
    ->setMatrix4fv("view"         ,glm::value_ptr(view    ))
    ->setMatrix4fv("proj"         ,glm::value_ptr(proj    ))
    ->set4fv      ("lightPosition",glm::value_ptr(lightPosition))
    ;

  glDrawArrays(GL_POINTS,0,nofTriangles);

  vao->unbind();

}

}
