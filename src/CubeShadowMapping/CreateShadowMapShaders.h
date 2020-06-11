#include <string>

#if 0
std::string const createShadowMapVertexShaderSource = R".(
layout(location=0)in vec3 position;
uniform float near = 0.1;
uniform float far  = 1000;
uniform vec4 lightPosition = vec4(0,0,0,1);
out int vInstanceID;
void main(){
  const mat4 views[6] = {
    mat4(vec4(+0,+0,-1,0), vec4(+0,-1,+0,0), vec4(-1,+0,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+0,+0,+1,0), vec4(+0,-1,+0,0), vec4(+1,+0,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+1,+0,+0,0), vec4(+0,+0,-1,0), vec4(+0,+1,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+1,+0,+0,0), vec4(+0,+0,+1,0), vec4(+0,-1,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+1,+0,+0,0), vec4(+0,-1,+0,0), vec4(+0,+0,-1,0), vec4(0,0,0,1)),
    mat4(vec4(-1,+0,+0,0), vec4(+0,-1,+0,0), vec4(+0,+0,+1,0), vec4(0,0,0,1))
  };

  mat4 projection = mat4(
    vec4(1,0,0,0),
    vec4(0,1,0,0),
    vec4(0,0,-(far+near)/(far-near),-1),
    vec4(0,0,-2*far*near/(far-near),0));
  gl_Position = projection*views[gl_InstanceID]*vec4(position-lightPosition.xyz,1);
  vInstanceID = gl_InstanceID;
}).";

std::string const createShadowMapGeometryShaderSource = R".(
layout(triangles)in;
layout(triangle_strip,max_vertices=3)out;
in int vInstanceID[];
void main(){
  gl_Layer = vInstanceID[0];
  gl_Position = gl_in[0].gl_Position;EmitVertex();
  gl_Position = gl_in[1].gl_Position;EmitVertex();
  gl_Position = gl_in[2].gl_Position;EmitVertex();
  EndPrimitive();
}).";

std::string const createShadowMapFragmentShaderSource = R".(
void main(){
}
).";
#endif

#if 1
std::string const createShadowMapVertexShaderSource = R".(
layout(location=0)in vec3 position;
uniform vec4 lightPosition = vec4(0,0,0,1);
void main(){
  gl_Position = vec4(position-lightPosition.xyz,1);
}).";

std::string const createShadowMapGeometryShaderSource = R".(
layout(triangles)in;
layout(triangle_strip,max_vertices=3*6)out;

uniform float near = 0.1;
uniform float far  = 1000;

int faceId(in vec3 v){
  vec3 a = abs(v);
  if(a.x>a.y){
    if(a.x>a.z)
      return int(v.x<0);
    else
      return int(v.z<0)+4;
  }else{
    if(a.y>a.z)
      return int(v.y<0)+2;
    else
      return int(v.z<0)+4;
  }
}

void main(){
  const mat4 views[6] = {
    mat4(vec4(+0,+0,-1,0), vec4(+0,-1,+0,0), vec4(-1,+0,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+0,+0,+1,0), vec4(+0,-1,+0,0), vec4(+1,+0,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+1,+0,+0,0), vec4(+0,+0,-1,0), vec4(+0,+1,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+1,+0,+0,0), vec4(+0,+0,+1,0), vec4(+0,-1,+0,0), vec4(0,0,0,1)),
    mat4(vec4(+1,+0,+0,0), vec4(+0,-1,+0,0), vec4(+0,+0,-1,0), vec4(0,0,0,1)),
    mat4(vec4(-1,+0,+0,0), vec4(+0,-1,+0,0), vec4(+0,+0,+1,0), vec4(0,0,0,1))
  };

  mat4 projection = mat4(
    vec4(1,0,0,0),
    vec4(0,1,0,0),
    vec4(0,0,-(far+near)/(far-near),-1),
    vec4(0,0,-2*far*near/(far-near),0));

  mat4 m;

  int f0 = faceId(gl_in[0].gl_Position.xyz);
  int f1 = faceId(gl_in[1].gl_Position.xyz);
  int f2 = faceId(gl_in[2].gl_Position.xyz);

  if(f0 == f1 && f0 == f2){
    gl_Layer = f0;
    m = projection*views[f0];
    gl_Position = m*gl_in[0].gl_Position;EmitVertex();
    gl_Position = m*gl_in[1].gl_Position;EmitVertex();
    gl_Position = m*gl_in[2].gl_Position;EmitVertex();
    EndPrimitive();
    return;
  }


  for(int i=0;i<6;++i){
    //if(gl_in[0].gl_Position.x < 0)continue;
    gl_Layer = i;
    m = projection*views[i];
    gl_Position = m*gl_in[0].gl_Position;EmitVertex();
    gl_Position = m*gl_in[1].gl_Position;EmitVertex();
    gl_Position = m*gl_in[2].gl_Position;EmitVertex();
    EndPrimitive();
  }
}).";
std::string const createShadowMapFragmentShaderSource = R".(
void main(){
}
).";
#endif
