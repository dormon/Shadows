#include <Sintorn/ShadowFrustaShaders.h>
#include <GLSLLine.h>

std::string const sintorn::shadowFrustaShader = 
GLSL_LINE
R".(
//"methods/Sintorn/shadowfrustum.comp";
//DO NOT EDIT ANYTHING BELOW THIS LINE

#ifndef BIAS
  #define BIAS 0.0001
#endif//BIAS

#ifndef WAVEFRONT_SIZE
  #define WAVEFRONT_SIZE 64
#endif//WAVEFRONT_SIZE

#ifndef VEC4_PER_SHADOWFRUSTUM
  #define VEC4_PER_SHADOWFRUSTUM 6
#endif//VEC4_PER_SHADOWFRUSTUM

#define FLOATS_PER_SHADOWFRUSTUM (VEC4_PER_SHADOWFRUSTUM*4)

layout(local_size_x=WAVEFRONT_SIZE)in;

layout(std430,binding=0)buffer Triangles   {vec4 triangles   [];};
layout(std430,binding=1)buffer ShadowFrusta{vec4 shadowFrusta[];};

uniform uint nofTriangles                       ;
uniform vec4 lightPosition                      ;
uniform mat4 modelViewProjection                ;
uniform mat4 transposeInverseModelViewProjection;

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

void main(){
	uint gid=gl_GlobalInvocationID.x;
  if(gid >= nofTriangles)return;
	vec3 v0=triangles[gid*3+0].xyz;
	vec3 v1=triangles[gid*3+1].xyz;
	vec3 v2=triangles[gid*3+2].xyz;

	vec4 e0 = getPlane(v0,v1,lightPosition);
	vec4 e1 = getPlane(v1,v2,lightPosition);
	vec4 e2 = getPlane(v2,v0,lightPosition);
	vec4 e3 = getPlane(
			v0 + BIAS*normalize(v0*lightPosition.w-lightPosition.xyz),
			v1 + BIAS*normalize(v1*lightPosition.w-lightPosition.xyz),
			v2 + BIAS*normalize(v2*lightPosition.w-lightPosition.xyz));
	bool backFacing=false;
	if(dot(e3,lightPosition)>0){
		backFacing=true;
		e0=-e0;
		e1=-e1;
		e2=-e2;
		e3=-e3;
	}
	e3=transposeInverseModelViewProjection*e3;
	
	vec4 LP   = modelViewProjection*lightPosition;
	vec4 V0CS = modelViewProjection*vec4(v0,1);
	vec4 V1CS = modelViewProjection*vec4(v1,1);
	vec4 V2CS = modelViewProjection*vec4(v2,1);
	vec3 sqedges[4];
	int nof_sqedges=0;
	if(sign(e0.z)!=sign(e2.z))sqedges[nof_sqedges++]=(e0.z<0?-1:1)*-cross(LP.xyw,V0CS.xyw);
	if(sign(e0.z)!=sign(e1.z))sqedges[nof_sqedges++]=(e0.z<0?-1:1)* cross(LP.xyw,V1CS.xyw);
	if(sign(e1.z)!=sign(e2.z))sqedges[nof_sqedges++]=(e1.z<0?-1:1)* cross(LP.xyw,V2CS.xyw);
	// If not backfacing culling
	if(backFacing)
		for(int i=0;i<nof_sqedges;i++) 
			sqedges[i]=-sqedges[i]; 
		
	shadowFrusta[gid*VEC4_PER_SHADOWFRUSTUM+0]=e0;
	shadowFrusta[gid*VEC4_PER_SHADOWFRUSTUM+1]=e1;
	shadowFrusta[gid*VEC4_PER_SHADOWFRUSTUM+2]=e2;
	shadowFrusta[gid*VEC4_PER_SHADOWFRUSTUM+3]=e3;
	shadowFrusta[gid*VEC4_PER_SHADOWFRUSTUM+4]=vec4(sqedges[0],nof_sqedges);
	shadowFrusta[gid*VEC4_PER_SHADOWFRUSTUM+5]=vec4(sqedges[1],1);
}).";


