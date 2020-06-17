#include <RSSV/getEdgePlanesShader.h>

std::string const rssv::getEdgePlanesShader = R".(

// this function computes plane in clip space from 3 points in clip space
vec4 getClipPlaneSkala_(in vec4 A,in vec4 B,in vec4 C){
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

bool less4(in vec4 a,in vec4 b){
  if(a.x < b.x)return true;
  if(a.x > b.x)return false;
  if(a.y < b.y)return true;
  if(a.y > b.y)return false;
  if(a.z < b.z)return true;
  if(a.z > b.z)return false;
  if(a.w < b.w)return true;
  if(a.w > b.w)return false;
  return false;
}

#if !defined(ORDERED_SKALA)
#error "define ORDERED_SKALA"
#endif

vec4 getClipPlaneSkala(in vec4 A,in vec4 B,in vec4 C){
#if ORDERED_SKALA == 1
  if(less4(A,B)){
    if(less4(B,C)){
      return getClipPlaneSkala_(A,B,C);
    }else{
      if(less4(A,C)){
        return -getClipPlaneSkala_(A,C,B);
      }else{
        return getClipPlaneSkala_(C,A,B);
      }
    }
  }else{
    if(less4(C,B)){
      return -getClipPlaneSkala_(C,B,A);
    }else{
      if(less4(C,A)){
        return getClipPlaneSkala_(B,C,A);
      }else{
        return -getClipPlaneSkala_(B,A,C);
      }
    }
  }
#else
 return getClipPlaneSkala_(A,B,C);
#endif
}

void getEdgePlanesSkala(
    out vec4 edgePlane     ,
    out vec4 aPlane        ,
    out vec4 bPlane        ,
    out vec4 abPlane       ,
    in  vec4 edgeAClipSpace,
    in  vec4 edgeBClipSpace,
    in  vec4 lightClipSpace){
  edgePlane = normalize( getClipPlaneSkala(edgeAClipSpace,edgeBClipSpace,lightClipSpace));
  aPlane    = normalize( getClipPlaneSkala(edgeAClipSpace,lightClipSpace,edgeAClipSpace+vec4(edgePlane.xyz,0)));
  bPlane    = normalize( getClipPlaneSkala(edgeBClipSpace,edgeBClipSpace+vec4(edgePlane.xyz,0),lightClipSpace));
  abPlane   = normalize(-getClipPlaneSkala(edgeAClipSpace,edgeBClipSpace,edgeAClipSpace-vec4(edgePlane.xyz,0)));
}

vec4 getClipPlane(in vec4 a,in vec4 b,in vec4 c){
  if(a.w==0){
    if(b.w==0){
      if(c.w==0){
        return vec4(0,0,0,normalize(cross(normalize(b.xyz-a.xyz),normalize(c.xyz-a.xyz))).z);
      }else{
        vec3 n = normalize(cross(normalize(a.xyz*c.w-c.xyz*a.w),normalize(b.xyz*c.w-c.xyz*b.w)));
        return vec4(n*c.w,-dot(n,c.xyz));
      }
    }else{
      vec3 n = normalize(cross(normalize(c.xyz*b.w-b.xyz*c.w),normalize(a.xyz*b.w-b.xyz*a.w)));
      return vec4(n*b.w,-dot(n,b.xyz));
    }
  }else{
    vec3 n = normalize(cross(normalize(b.xyz*a.w-a.xyz*b.w),normalize(c.xyz*a.w-a.xyz*c.w)));
    return vec4(n*a.w,-dot(n,a.xyz));
  }
}

void getEdgePlanes(
    out vec4 edgePlane     ,
    out vec4 aPlane        ,
    out vec4 bPlane        ,
    out vec4 abPlane       ,
    in  vec4 edgeAClipSpace,
    in  vec4 edgeBClipSpace,
    in  vec4 lightClipSpace){
  edgePlane = getClipPlane(edgeAClipSpace,edgeBClipSpace,lightClipSpace);
  
  vec3 an = normalize(cross(
        edgePlane.xyz,
        normalize(edgeAClipSpace.xyz*lightClipSpace.w-lightClipSpace.xyz*edgeAClipSpace.w)));
  aPlane = vec4(an*abs(edgeAClipSpace.w),-dot(an,edgeAClipSpace.xyz)*sign(edgeAClipSpace.w));
  
  vec3 bn = normalize(cross(
        normalize(edgeBClipSpace.xyz*lightClipSpace.w-lightClipSpace.xyz*edgeBClipSpace.w),
        edgePlane.xyz));
  bPlane = vec4(bn*abs(edgeBClipSpace.w),-dot(bn,edgeBClipSpace.xyz)*sign(edgeBClipSpace.w));
  
  vec3 abn = normalize(cross(
        normalize(edgeBClipSpace.xyz*edgeAClipSpace.w-edgeAClipSpace.xyz*edgeBClipSpace.w),
        edgePlane.xyz));
  abPlane = vec4(abn*abs(edgeAClipSpace.w),-dot(abn,edgeAClipSpace.xyz)*sign(edgeAClipSpace.w));
}


).";
