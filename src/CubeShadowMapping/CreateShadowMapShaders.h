#include <string>

#if 1
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

#if 0
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

vec4 getClipPlaneSkala(in vec4 A,in vec4 B,in vec4 C){
  float x1 = A.x;
  float x2 = B.x;
  float x3 = C.x;
  float y1 = A.y;
  float y2 = B.y;
  float y3 = C.y;
  float z1 = A.z;
  float z2 = B.z;
  float z3 = C.z;
  float w1 = A.w;
  float w2 = B.w;
  float w3 = C.w;

  float a =  y1*(z2*w3-z3*w2) - y2*(z1*w3-z3*w1) + y3*(z1*w2-z2*w1);
  float b = -x1*(z2*w3-z3*w2) + x2*(z1*w3-z3*w1) - x3*(z1*w2-z2*w1);
  float c =  x1*(y2*w3-y3*w2) - x2*(y1*w3-y3*w1) + x3*(y1*w2-y2*w1);
  float d = -x1*(y2*z3-y3*z2) + x2*(y1*z3-y3*z1) - x3*(y1*z2-y2*z1);
  return vec4(a,b,c,d);
}

vec3 orderedCross(in vec3 a,in vec3 b){
  if(a.x < b.x)return+cross(a,b);
  if(a.x > b.x)return-cross(b,a);
  if(a.y < b.y)return+cross(a,b);
  if(a.y > b.y)return-cross(b,a);
  if(a.z < b.z)return+cross(a,b);
  if(a.z > b.z)return-cross(b,a);
  return cross(a,b);
}

void computeConservative(out vec4 af,out vec4 bf,out vec4 cf,in vec4 a,in vec4 b,in vec4 c,in vec3 mask,in vec2 hPixel){
  vec4 plane = getClipPlaneSkala(a,b,c);
  plane /= length(plane.xyz);

  //if(abs((-2*near*far)/(far-near)*plane.z) < 0.2000199){
  //  af = vec4(0,0,0,1);
  //  bf = vec4(0,0,0,1);
  //  cf = vec4(0,0,0,1);
  //  return;
  //}

  //vec3 pa = cross(a.xyw-c.xyw,c.xyw);
  //vec3 pb = cross(b.xyw-a.xyw,a.xyw);
  //vec3 pc = cross(c.xyw-b.xyw,b.xyw);
  vec3 pa = orderedCross(a.xyw,c.xyw);
  vec3 pb = orderedCross(b.xyw,a.xyw);
  vec3 pc = orderedCross(c.xyw,b.xyw);


  pa.z -= dot(hPixel,abs(pa.xy))*mask.x;
  pb.z -= dot(hPixel,abs(pb.xy))*mask.y;
  pc.z -= dot(hPixel,abs(pc.xy))*mask.z;

  af.xyw = cross(pa,pb);
  bf.xyw = cross(pb,pc);
  cf.xyw = cross(pc,pa);

  af.z = -dot(af.xyw,plane.xyw) / plane.z;
  bf.z = -dot(bf.xyw,plane.xyw) / plane.z;
  cf.z = -dot(cf.xyw,plane.xyw) / plane.z;

}

vec4 plane(in vec3 a,in vec3 b,in vec3 c){
  vec3 n = normalize(cross(normalize(b-a),normalize(c-a)));
  return vec4(n,-dot(n,a));
}

void com(out vec3 af,out vec3 bf,out vec3 cf,in vec3 a,in vec3 b,in vec3 c,in vec3 mask,in vec2 hPixel){
  vec2 A = a.xy/a.z;
  vec2 B = b.xy/b.z;
  vec2 C = c.xy/c.z;
  float ee = hPixel.x*hPixel.y*4*40;
  if(length(cross(vec3(B-A,0),vec3(C-A,0)))<ee){
    af = a;
    bf = a;
    cf = a;
    return;
  }
  //if(abs(pp.w) < 200){
  //  af = a;
  //  bf = a;
  //  cf = a;
  //  return;
  //}

  vec4 pp = plane(a,b,c);
  vec3 pa = orderedCross(a,c);
  vec3 pb = orderedCross(b,a);
  vec3 pc = orderedCross(c,b);

  pa.z += dot(hPixel,abs(pa.xy))*mask.x;
  pb.z += dot(hPixel,abs(pb.xy))*mask.y;
  pc.z += dot(hPixel,abs(pc.xy))*mask.z;

  af = orderedCross(pa,pb);
  bf = orderedCross(pb,pc);
  cf = orderedCross(pc,pa);


  af *= abs(-pp.w/dot(pp.xyz,af));
  bf *= abs(-pp.w/dot(pp.xyz,bf));
  cf *= abs(-pp.w/dot(pp.xyz,cf));
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

  uint tri = uint(gl_PrimitiveID);

  if(f0 == f1 && f0 == f2){
    gl_Layer = f0;
    m = projection*views[f0];
 
    //vec4 tp = plane(vec3(views[f0]*gl_in[0].gl_Position),vec3(views[f0]*gl_in[1].gl_Position),vec3(views[f0]*gl_in[2].gl_Position));
    //if(abs(tp.w)<200)return;


    //vec4 a = m*gl_in[0].gl_Position;
    //vec4 b = m*gl_in[1].gl_Position;
    //vec4 c = m*gl_in[2].gl_Position;
    //vec4 fa;
    //vec4 fb;
    //vec4 fc;
    //computeConservative(fa,fb,fc,a,b,c,vec3(1),vec2(1/1024.));
    
    vec3 aa = vec3(views[f0]*gl_in[0].gl_Position);
    vec3 bb = vec3(views[f0]*gl_in[1].gl_Position);
    vec3 cc = vec3(views[f0]*gl_in[2].gl_Position);

    vec4 fa;
    vec4 fb;
    vec4 fc;
    //if((tri/100)%1==0)
    //  com(fa.xyz,fb.xyz,fc.xyz,aa,bb,cc,vec3(-1,0,0),vec2(1/1024.));
    //else
      com(fa.xyz,fb.xyz,fc.xyz,aa,bb,cc,vec3(0,0,0),vec2(1/1024.));
    fa.w=1;
    fb.w=1;
    fc.w=1;
    fa = projection*fa;
    fb = projection*fb;
    fc = projection*fc;

    gl_Position = fa;EmitVertex();
    gl_Position = fb;EmitVertex();
    gl_Position = fc;EmitVertex();
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
