#include <RSSV/shadowFrustaShader.h>


std::string const rssv::shadowFrusta::fwdShader = R".(
void computeShadowFrustaJOB();

#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef NOF_TRIANGLES
#define NOF_TRIANGLES 0u
#endif//NOF_TRIANGLES

#ifndef WGS
#define WGS 64
#endif//WGS

#ifndef TRIANGLE_ALIGNMENT
#define TRIANGLE_ALIGNMENT 128
#endif//TRIANGLE_ALIGNMENT

#ifndef SF_ALIGNMENT
#define SF_ALIGNMENT 128
#endif//SF_ALIGNMENT

#ifndef BIAS
#define BIAS 1.0f
#endif//BIAS

#ifndef SF_INTERLEAVE
#define SF_INTERLEAVE 0
#endif//SF_INTERLEAVE

#ifndef TRIANGLE_INTERLEAVE
#define TRIANGLE_INTERLEAVE 1
#endif//TRIANGLE_INTERLEAVE

#ifndef VEC4_PER_SHADOWFRUSTUM
#define VEC4_PER_SHADOWFRUSTUM 4
#endif//VEC4_PER_SHADOWFRUSTUM

#ifndef MORE_PLANES
#define MORE_PLANES 0
#endif//MORE_PLANES

#ifndef EXACT_TRIANGLE_AABB
#define EXACT_TRIANGLE_AABB 0
#endif//EXACT_TRIANGLE_AABB

#if WGS > WARP

#if !defined(SHARED_MEMORY_SIZE) || (SHARED_MEMORY_SIZE) < 1
#undef SHARED_MEMORY_SIZE
#define SHARED_MEMORY_SIZE 1
#endif

#define shadowFrustaJobStartO 0
#define shadowFrustaJobStart getShared1u(shadowFrustaJobStartO)

#endif

).";


std::string const rssv::shadowFrusta::computeShader = R".(


const uint planesPerSF = 4u + MORE_PLANES*3u + EXACT_TRIANGLE_AABB*3u;
const uint floatsPerPlane = 4u;
const uint floatsPerSF = planesPerSF * floatsPerPlane;

#line 38

const uint alignedNofTriangles = (uint(NOF_TRIANGLES / TRIANGLE_ALIGNMENT) + uint((NOF_TRIANGLES % TRIANGLE_ALIGNMENT) != 0u)) * TRIANGLE_ALIGNMENT;
const uint alignedNofSF        = (uint(NOF_TRIANGLES /       SF_ALIGNMENT) + uint((NOF_TRIANGLES %       SF_ALIGNMENT) != 0u)) *       SF_ALIGNMENT;


uniform vec4 lightPosition                      ;
uniform mat4 transposeInverseModelViewProjection;

#if EXACT_TRIANGLE_AABB == 1
uniform mat4 projView;
#endif

int greaterVec(vec3 a,vec3 b){
	return int(dot(ivec3(sign(a-b)),ivec3(4,2,1)));
}

vec4 getOrderedPlane(vec3 A,vec3 B,vec4 L){
	if(greaterVec(A,B)>0){
		vec3 n=normalize(cross(A-B,L.xyz-B*L.w));
		return vec4(-n,dot(n,B));
	}else{
		vec3 n=normalize(cross(B-A,L.xyz-A*L.w));
		return vec4(n,-dot(n,A));
	}
}

vec4 getPlane(vec3 A,vec3 B,vec4 L){
	return transposeInverseModelViewProjection*getOrderedPlane(A,B,L);
}

vec4 getPlane(vec3 A,vec3 B,vec3 C){
	vec3 n=normalize(cross(B-A,C-A));
	return vec4(n,-dot(n,A));
}

void computeShadowFrusta(uint gid){
  vec3 v0;
  vec3 v1;
  vec3 v2;

#if TRIANGLE_INTERLEAVE == 1
  v0[0] = triangles[alignedNofTriangles*0u + gid];
  v0[1] = triangles[alignedNofTriangles*1u + gid];
  v0[2] = triangles[alignedNofTriangles*2u + gid];
  v1[0] = triangles[alignedNofTriangles*3u + gid];
  v1[1] = triangles[alignedNofTriangles*4u + gid];
  v1[2] = triangles[alignedNofTriangles*5u + gid];
  v2[0] = triangles[alignedNofTriangles*6u + gid];
  v2[1] = triangles[alignedNofTriangles*7u + gid];
  v2[2] = triangles[alignedNofTriangles*8u + gid];
#else
  v0[0] = triangles[gid*9u+0u];
  v0[1] = triangles[gid*9u+1u];
  v0[2] = triangles[gid*9u+2u];
  v1[0] = triangles[gid*9u+3u];
  v1[1] = triangles[gid*9u+4u];
  v1[2] = triangles[gid*9u+5u];
  v2[0] = triangles[gid*9u+6u];
  v2[1] = triangles[gid*9u+7u];
  v2[2] = triangles[gid*9u+8u];
#endif

	vec4 e0 = getPlane(v0,v1,lightPosition);
	vec4 e1 = getPlane(v1,v2,lightPosition);
	vec4 e2 = getPlane(v2,v0,lightPosition);
	vec4 e3 = getPlane(
			v0 + BIAS*normalize(v0*lightPosition.w-lightPosition.xyz),
			v1 + BIAS*normalize(v1*lightPosition.w-lightPosition.xyz),
			v2 + BIAS*normalize(v2*lightPosition.w-lightPosition.xyz));


#if MORE_PLANES == 1
  vec4 f0;
  vec4 f1;
  vec4 f2;
  vec3 l0 = normalize(lightPosition.xyz-v0*lightPosition.w);
  vec3 l1 = normalize(lightPosition.xyz-v1*lightPosition.w);
  vec3 l2 = normalize(lightPosition.xyz-v2*lightPosition.w);
  f0.xyz = -normalize(cross(cross(l0,normalize(v0-v1)+normalize(v0-v2)),l0));
  f1.xyz = -normalize(cross(cross(l1,normalize(v1-v0)+normalize(v1-v2)),l1));
  f2.xyz = -normalize(cross(cross(l2,normalize(v2-v0)+normalize(v2-v1)),l2));
  f0.w = -dot(f0.xyz,v0);
  f1.w = -dot(f1.xyz,v1);
  f2.w = -dot(f2.xyz,v2);
  f0 = transposeInverseModelViewProjection*f0;
  f1 = transposeInverseModelViewProjection*f1;
  f2 = transposeInverseModelViewProjection*f2;
#endif

	float ffc = 1.f;
	if(dot(e3,lightPosition)>0){
		ffc=0.f;
		e0=-e0;
		e1=-e1;
		e2=-e2;
		e3=-e3;
	}
	e3=transposeInverseModelViewProjection*e3;

#if EXACT_TRIANGLE_AABB == 1
  vec4 t0 = projView * vec4(v0,1);
  vec4 t1 = projView * vec4(v1,1);
  vec4 t2 = projView * vec4(v2,1);
#endif



#if SF_INTERLEAVE == 1
  shadowFrusta[alignedNofSF* 0u + gid] = e0[0];
  shadowFrusta[alignedNofSF* 1u + gid] = e0[1];
  shadowFrusta[alignedNofSF* 2u + gid] = e0[2];
  shadowFrusta[alignedNofSF* 3u + gid] = e0[3];

  shadowFrusta[alignedNofSF* 4u + gid] = e1[0];
  shadowFrusta[alignedNofSF* 5u + gid] = e1[1];
  shadowFrusta[alignedNofSF* 6u + gid] = e1[2];
  shadowFrusta[alignedNofSF* 7u + gid] = e1[3];

  shadowFrusta[alignedNofSF* 8u + gid] = e2[0];
  shadowFrusta[alignedNofSF* 9u + gid] = e2[1];
  shadowFrusta[alignedNofSF*10u + gid] = e2[2];
  shadowFrusta[alignedNofSF*11u + gid] = e2[3];

  shadowFrusta[alignedNofSF*12u + gid] = e3[0];
  shadowFrusta[alignedNofSF*13u + gid] = e3[1];
  shadowFrusta[alignedNofSF*14u + gid] = e3[2];
  shadowFrusta[alignedNofSF*15u + gid] = e3[3];

  #if MORE_PLANES == 1
    shadowFrusta[alignedNofSF*16u + gid] = f0[0];
    shadowFrusta[alignedNofSF*17u + gid] = f0[1];
    shadowFrusta[alignedNofSF*18u + gid] = f0[2];
    shadowFrusta[alignedNofSF*19u + gid] = f0[3];

    shadowFrusta[alignedNofSF*20u + gid] = f1[0];
    shadowFrusta[alignedNofSF*21u + gid] = f1[1];
    shadowFrusta[alignedNofSF*22u + gid] = f1[2];
    shadowFrusta[alignedNofSF*23u + gid] = f1[3];

    shadowFrusta[alignedNofSF*24u + gid] = f2[0];
    shadowFrusta[alignedNofSF*25u + gid] = f2[1];
    shadowFrusta[alignedNofSF*26u + gid] = f2[2];
    shadowFrusta[alignedNofSF*27u + gid] = f2[3];
  #endif

  #if EXACT_TRIANGLE_AABB == 1 && MORE_PLANES == 1
    shadowFrusta[alignedNofSF*28u + gid] = t0[0];
    shadowFrusta[alignedNofSF*29u + gid] = t0[1];
    shadowFrusta[alignedNofSF*30u + gid] = t0[2];
    shadowFrusta[alignedNofSF*31u + gid] = t0[3];

    shadowFrusta[alignedNofSF*32u + gid] = t1[0];
    shadowFrusta[alignedNofSF*33u + gid] = t1[1];
    shadowFrusta[alignedNofSF*34u + gid] = t1[2];
    shadowFrusta[alignedNofSF*35u + gid] = t1[3];

    shadowFrusta[alignedNofSF*36u + gid] = t2[0];
    shadowFrusta[alignedNofSF*37u + gid] = t2[1];
    shadowFrusta[alignedNofSF*38u + gid] = t2[2];
    shadowFrusta[alignedNofSF*39u + gid] = t2[3];
  #endif

  #if EXACT_TRIANGLE_AABB == 1 && MORE_PLANES == 0
    shadowFrusta[alignedNofSF*16u + gid] = t0[0];
    shadowFrusta[alignedNofSF*17u + gid] = t0[1];
    shadowFrusta[alignedNofSF*18u + gid] = t0[2];
    shadowFrusta[alignedNofSF*19u + gid] = t0[3];

    shadowFrusta[alignedNofSF*20u + gid] = t1[0];
    shadowFrusta[alignedNofSF*21u + gid] = t1[1];
    shadowFrusta[alignedNofSF*22u + gid] = t1[2];
    shadowFrusta[alignedNofSF*23u + gid] = t1[3];

    shadowFrusta[alignedNofSF*24u + gid] = t2[0];
    shadowFrusta[alignedNofSF*25u + gid] = t2[1];
    shadowFrusta[alignedNofSF*26u + gid] = t2[2];
    shadowFrusta[alignedNofSF*27u + gid] = t2[3];
  #endif


#else
  shadowFrusta[gid*floatsPerSF+ 0u] = e0[0];
  shadowFrusta[gid*floatsPerSF+ 1u] = e0[1];
  shadowFrusta[gid*floatsPerSF+ 2u] = e0[2];
  shadowFrusta[gid*floatsPerSF+ 3u] = e0[3];
  shadowFrusta[gid*floatsPerSF+ 4u] = e1[0];
  shadowFrusta[gid*floatsPerSF+ 5u] = e1[1];
  shadowFrusta[gid*floatsPerSF+ 6u] = e1[2];
  shadowFrusta[gid*floatsPerSF+ 7u] = e1[3];
  shadowFrusta[gid*floatsPerSF+ 8u] = e2[0];
  shadowFrusta[gid*floatsPerSF+ 9u] = e2[1];
  shadowFrusta[gid*floatsPerSF+10u] = e2[2];
  shadowFrusta[gid*floatsPerSF+11u] = e2[3];
  shadowFrusta[gid*floatsPerSF+12u] = e3[0];
  shadowFrusta[gid*floatsPerSF+13u] = e3[1];
  shadowFrusta[gid*floatsPerSF+14u] = e3[2];
  shadowFrusta[gid*floatsPerSF+15u] = e3[3];

  #if MORE_PLANES == 1
    shadowFrusta[gid*floatsPerSF+16u] = f0[0];
    shadowFrusta[gid*floatsPerSF+17u] = f0[1];
    shadowFrusta[gid*floatsPerSF+18u] = f0[2];
    shadowFrusta[gid*floatsPerSF+19u] = f0[3];
    shadowFrusta[gid*floatsPerSF+20u] = f1[0];
    shadowFrusta[gid*floatsPerSF+21u] = f1[1];
    shadowFrusta[gid*floatsPerSF+22u] = f1[2];
    shadowFrusta[gid*floatsPerSF+23u] = f1[3];
    shadowFrusta[gid*floatsPerSF+24u] = f2[0];
    shadowFrusta[gid*floatsPerSF+25u] = f2[1];
    shadowFrusta[gid*floatsPerSF+26u] = f2[2];
    shadowFrusta[gid*floatsPerSF+27u] = f2[3];
  #endif

  #if MORE_PLANES == 1 && EXACT_TRIANGLE_AABB == 1
    shadowFrusta[gid*floatsPerSF+28u] = t0[0];
    shadowFrusta[gid*floatsPerSF+29u] = t0[1];
    shadowFrusta[gid*floatsPerSF+30u] = t0[2];
    shadowFrusta[gid*floatsPerSF+31u] = t0[3];
    shadowFrusta[gid*floatsPerSF+32u] = t1[0];
    shadowFrusta[gid*floatsPerSF+33u] = t1[1];
    shadowFrusta[gid*floatsPerSF+34u] = t1[2];
    shadowFrusta[gid*floatsPerSF+35u] = t1[3];
    shadowFrusta[gid*floatsPerSF+36u] = t2[0];
    shadowFrusta[gid*floatsPerSF+37u] = t2[1];
    shadowFrusta[gid*floatsPerSF+38u] = t2[2];
    shadowFrusta[gid*floatsPerSF+39u] = t2[3];
  #endif
  
  #if MORE_PLANES == 0 && EXACT_TRIANGLE_AABB == 1
    shadowFrusta[gid*floatsPerSF+16u] = v0[0];
    shadowFrusta[gid*floatsPerSF+17u] = v0[1];
    shadowFrusta[gid*floatsPerSF+18u] = v0[2];
    shadowFrusta[gid*floatsPerSF+19u] = v0[3];
    shadowFrusta[gid*floatsPerSF+20u] = v1[0];
    shadowFrusta[gid*floatsPerSF+21u] = v1[1];
    shadowFrusta[gid*floatsPerSF+22u] = v1[2];
    shadowFrusta[gid*floatsPerSF+23u] = v1[3];
    shadowFrusta[gid*floatsPerSF+24u] = v2[0];
    shadowFrusta[gid*floatsPerSF+25u] = v2[1];
    shadowFrusta[gid*floatsPerSF+26u] = v2[2];
    shadowFrusta[gid*floatsPerSF+27u] = v2[3];
  #endif

#endif
}

void computeShadowFrustaJOB(){
#if USE_PERSISTENT_THREADS == 1
  uint job;
  for(;;){
    if(gl_LocalInvocationIndex == 0)
      job = atomicAdd(shadowFrustaJobCounter,gl_WorkGroupSize.x);

#if WGS > WARP
    if(gl_LocalInvocationIndex == 0)
      toShared1u(shadowFrustaJobStartO,job);
    //memoryBarrierShared();
    barrier();
    job = shadowFrustaJobStart;
#else
    job = readFirstInvocationARB(job);
#endif

    if(job+gl_LocalInvocationIndex >= NOF_TRIANGLES)break;

    computeShadowFrusta(job+gl_LocalInvocationIndex);

  }
#else
	uint gid=gl_GlobalInvocationID.x;
  if(gid >= NOF_TRIANGLES)return;
  computeShadowFrusta(gid);
#endif
}





).";


std::string const rssv::shadowFrusta::mainShader = R".(
layout(local_size_x=WGS)in;

layout(std430,binding=0)buffer Triangles   {float triangles   [];};
layout(std430,binding=1)buffer ShadowFrusta{float shadowFrusta[];};

#if USE_PERSISTENT_THREADS == 1
layout(std430,binding=2)buffer ShadowFrustaJobCounter  {uint  shadowFrustaJobCounter ;};
#endif


void main(){
  computeShadowFrustaJOB();
}
).";





