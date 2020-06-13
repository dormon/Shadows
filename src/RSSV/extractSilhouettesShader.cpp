#include <RSSV/extractSilhouettesShader.h>

std::string const rssv::extractSilhouettesShader = R".(
#line 4


#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef MAX_MULTIPLICITY
#define MAX_MULTIPLICITY 2
#endif//MAX_MULTIPLICITY

#ifndef WORKGROUP_SIZE_X
#define WORKGROUP_SIZE_X 64
#endif//WORKGROUP_SIZE_X

#ifndef ALIGN_SIZE
#define ALIGN_SIZE 128
#endif//ALIGN_SIZE

#ifndef NOF_EDGES
#define NOF_EDGES 0
#endif//NOF_EDGES

layout(local_size_x=WORKGROUP_SIZE_X)in;

layout(std430,binding=0)readonly buffer EdgePlanes         {float edgePlanes [];};

layout(std430,binding=3)buffer MultBuffer{
  uint nofSilhouettes;
  uint multBuffer[];
};

#if COMPUTE_SILHOUETTE_PLANES == 1
layout(std430,binding=4)buffer SilhouettePlanes{
  float silhouettePlanes[];
};
uniform mat4 invTran          ;
uniform mat4 projView         ;
#endif

uniform vec4 lightPosition = vec4(100,100,100,1);
uniform mat4 mvp           = mat4(1)            ;

shared uint localCounter;
shared uint globalOffset;

void storeSilhouettePlanes(in uint wh,in uint edge,in int mult){
#if COMPUTE_SILHOUETTE_PLANES == 1
  vec3 edgeA;
  vec3 edgeB;
  loadEdge(edgeA,edgeB,edge);

  vec3 n = normalize(cross(edgeB-edgeA,lightPosition.xyz-edgeA));
  vec4 edgePlane = invTran*vec4(n  ,-dot(n  ,edgeA));

  vec3 an = normalize(cross(n,edgeA-lightPosition.xyz));
  vec4 aPlane    = invTran*vec4(an ,-dot(an ,edgeA));

  vec3 bn = normalize(cross(edgeB-lightPosition.xyz,n));
  vec4 bPlane     = invTran*vec4(bn ,-dot(bn ,edgeB));

  vec3 abn = normalize(cross(edgeB-edgeA,n));
  vec4 abPlane    = invTran*vec4(abn,-dot(abn,edgeA));

#if COMPUTE_SILHOUETTE_BRIDGES == 1 || COMPUTE_TRIANGLE_BRIDGES == 1 || EXACT_SILHOUETTE_AABB == 1
  vec4 edgeAClipSpace = projView*vec4(edgeA,1.f);
  vec4 edgeBClipSpace = projView*vec4(edgeB,1.f);
  const uint floatsPerSilhouette = 4*6+1;
#else
  const uint floatsPerSilhouette = 4*4+1;
#endif


  silhouettePlanes[wh*floatsPerSilhouette+0+0*4+0] = edgePlane[0]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+0*4+1] = edgePlane[1]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+0*4+2] = edgePlane[2]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+0*4+3] = edgePlane[3]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+1*4+0] =    aPlane[0]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+1*4+1] =    aPlane[1]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+1*4+2] =    aPlane[2]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+1*4+3] =    aPlane[3]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+2*4+0] =    bPlane[0]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+2*4+1] =    bPlane[1]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+2*4+2] =    bPlane[2]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+2*4+3] =    bPlane[3]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+3*4+0] =   abPlane[0]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+3*4+1] =   abPlane[1]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+3*4+2] =   abPlane[2]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+3*4+3] =   abPlane[3]; 

#if COMPUTE_SILHOUETTE_BRIDGES == 1 || COMPUTE_TRIANGLE_BRIDGES == 1 || EXACT_SILHOUETTE_AABB == 1
  silhouettePlanes[wh*floatsPerSilhouette+0+4*4+0] =   edgeAClipSpace[0]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+4*4+1] =   edgeAClipSpace[1]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+4*4+2] =   edgeAClipSpace[2]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+4*4+3] =   edgeAClipSpace[3]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+5*4+0] =   edgeBClipSpace[0]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+5*4+1] =   edgeBClipSpace[1]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+5*4+2] =   edgeBClipSpace[2]; 
  silhouettePlanes[wh*floatsPerSilhouette+0+5*4+3] =   edgeBClipSpace[3]; 
#endif

#if COMPUTE_SILHOUETTE_BRIDGES == 1 || COMPUTE_TRIANGLE_BRIDGES == 1 || EXACT_SILHOUETTE_AABB == 1
  silhouettePlanes[wh*floatsPerSilhouette+6*4+0] = float(mult); 
#else
  silhouettePlanes[wh*floatsPerSilhouette+4*4+0] = float(mult); 
#endif

#endif
}

void main(){

  if(gl_LocalInvocationID.x==0){
    localCounter = 0;
    globalOffset = 0;
  }
  barrier();

  uint gid = gl_GlobalInvocationID.x;
  if(gid>=NOF_EDGES)return;

  precise int Multiplicity=0;

  for(uint m=0;m<MAX_MULTIPLICITY;++m){
    vec4 plane;
    plane.x = edgePlanes[gid+ALIGN_OFFSET(6+m*4+0)];
    plane.y = edgePlanes[gid+ALIGN_OFFSET(6+m*4+1)];
    plane.z = edgePlanes[gid+ALIGN_OFFSET(6+m*4+2)];
    plane.w = edgePlanes[gid+ALIGN_OFFSET(6+m*4+3)];
    Multiplicity += int(sign(dot(plane,lightPosition)));
  }

  uint localOffset = atomicAdd(localCounter,uint(Multiplicity!=0));

  #if WORKGROUP_SIZE_X > WARP
    barrier();
  #endif

  if(gl_LocalInvocationID.x==0){
    globalOffset = atomicAdd(nofSilhouettes,localCounter);
  }

  #if WORKGROUP_SIZE_X > WARP
    barrier();
  #endif

  uint WH = globalOffset + localOffset;
  
#if COMPUTE_SILHOUETTE_PLANES != 1
  if(Multiplicity != 0){
    uint res = 0;
    res |= uint(Multiplicity << 29);
    res |= uint(gl_GlobalInvocationID.x);
    multBuffer[WH] = res;
  }
#endif

  if(Multiplicity != 0)
    storeSilhouettePlanes(WH,gid,Multiplicity);

}
).";
