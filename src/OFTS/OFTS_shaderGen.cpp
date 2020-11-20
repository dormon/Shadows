#include "OFTS_shaderGen.h"

#include <geGL/Program.h>
#include <geGL/Shader.h>

#include <fstream>
#include <algorithm>

using namespace ge;
using namespace gl;

namespace OFTS
{

const char* heatmapCS = R".(
layout(local_size_x = WG_SIZE) in;

layout(binding = 0, r32ui) uniform coherent uimage2DArray heatMap;

layout(binding=0) uniform sampler2D posTexture;
layout(binding=1) uniform sampler2D normalTexture;

uniform uvec2 screenResolution;
uniform uvec2 heatmapResolution;
uniform vec3  lightPos;
uniform mat4  lightVP[6];
uniform uint  frustumMask;

struct S_MinMax
{
	uvec2 minVal;
	uvec2 maxVal;
};

void main()
{
	const uint gid = gl_GlobalInvocationID.x;
	const uint lid = gl_LocalInvocationID.x;
	
	bool hasValidSample = true;
	
	if(gid >= (screenResolution.x * screenResolution.y))
    {
        hasValidSample = false;
    }
	
	const ivec2 coords = ivec2(gid % screenResolution.x, gid / screenResolution.x);
	
	//Read the position sample
	vec4 worldFragPos = texelFetch(posTexture, coords, 0);
	vec3 worldNormal = texelFetch(normalTexture, coords, 0).xyz;
	
	const float NdL = dot(worldNormal, normalize(lightPos - worldFragPos.xyz));
	
	// Early N dot L reject - this fragment is shadowed by lighting implicitly
	if(worldFragPos.w == 0.f || NdL < 0)
	{
		hasValidSample = false;
	}
	
	if(!hasValidSample) return;
	
	for(uint f = 0; f<6; ++f)
	{
		if(((frustumMask >> f) & 1)==0)
		{
			continue;
		}
		
		vec4 lightProjPos = lightVP[f] * worldFragPos;
		
		// If inside light frustum, add the fragment to the list
		const float w = lightProjPos.w;
		const bool isInsideFrustum = lightProjPos.x <= w && lightProjPos.x >= -w && lightProjPos.y <= w && lightProjPos.y >=-w && lightProjPos.z <= w && lightProjPos.z >= -w;
		const bool goodSample = hasValidSample && isInsideFrustum;
		
		//... to texture space
		lightProjPos /= w;
		lightProjPos.xyz = 0.5f * lightProjPos.xyz + 0.5f;
		
		//... to light space coords
		const ivec3 lightSpaceCoords = ivec3(lightProjPos.xy * heatmapResolution, f);
		
		//Add to heat map if valid
		if(goodSample)
		{
			imageAtomicAdd(heatMap, lightSpaceCoords, 1u);
		}
	}
}
).";

const char* matrixCS = R".(
#version 450 core

layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0, r32ui) uniform readonly uimage2DArray heatMap;

layout(packed, binding = 0) restrict writeonly buffer _proj
{ 
	mat4 lightP[12];
	uint nofMatrices[6];
};

uniform uvec2 heatmapResolution;
uniform vec4 frustumParams; //one is enough, as all frusta are the same (and it's in view space)
uniform uint treshold;
uniform uint frustumMask;

struct S_MinMax
{
	uvec2 minVal;
	uvec2 maxVal;
};

shared S_MinMax shExtents[2];
shared uint shUseReprojection;   

mat4 GetProjectionMatrix(in uint index)
{
	const float dw = frustumParams.x / float(heatmapResolution.x);
	const float dh = frustumParams.y / float(heatmapResolution.y);
	
	const uint minX = shExtents[index].minVal.x;
	const uint minY = shExtents[index].minVal.y;
	const uint maxX = shExtents[index].maxVal.x;
	const uint maxY = shExtents[index].maxVal.y;
	
	const vec2 minPoint = vec2(-0.5f*frustumParams.xy) + vec2(dw, dh) * vec2(minX, minY);
	const vec2 maxPoint = vec2(-0.5f*frustumParams.xy) + vec2(dw, dh) * vec2(maxX+1, maxY+1);
	
	const float nearZ = frustumParams.z;
	const float farZ = frustumParams.w;
	
	mat4 proj = mat4(0);
	proj[0][0] = 2*nearZ / (maxPoint.x - minPoint.x);
	proj[1][1] = 2*nearZ / (maxPoint.y - minPoint.y);
	proj[2][0] = (maxPoint.x + minPoint.x) / (maxPoint.x - minPoint.x);
	proj[2][1] = (maxPoint.y + minPoint.y) / (maxPoint.y - minPoint.y);
	proj[2][2] = -(farZ + nearZ)/(farZ - nearZ);
	proj[2][3] = -1;
	proj[3][2] = -(2.f * farZ * nearZ) / (farZ - nearZ);

	return proj;
}

//Runs in 6 workgroups
void main()
{
	const uvec2 lid = gl_LocalInvocationID.xy;
	const uint lIndex = gl_LocalInvocationIndex;
	const uint wgid = gl_WorkGroupID.x;
	
	if(((frustumMask >> wgid)&1u)==0)
	{
		if(lIndex==0)
		{
			nofMatrices[wgid] = 0;
		}
		
		return;
	}
	//*/
	if(lIndex==0)
	{
		shExtents[0].minVal = shExtents[1].minVal = heatmapResolution - uvec2(1, 1);
		shExtents[0].maxVal = shExtents[1].maxVal = uvec2(0, 0);
		shUseReprojection = 0;
	}
	
	barrier();
	
	const uvec2 blockSize = heatmapResolution / 32;
	
	const uvec2 start = lid * blockSize;
	const uvec2 end = start + blockSize;
	
	S_MinMax minMax[2];
	minMax[0].minVal = minMax[1].minVal = uvec2(heatmapResolution - uvec2(1, 1));
	minMax[0].maxVal = minMax[1].maxVal = uvec2(0, 0);
	uint useReprojection = 0;
	
	for(uint x = start.x; x < end.x; ++x)
	{
		for(uint y = start.y; y < end.y; ++y)
		{
			const uint len = imageLoad(heatMap, ivec3(x, y, wgid)).x;
			if(len > 0)
			{
				minMax[0].minVal = min(minMax[0].minVal, uvec2(x, y));
				minMax[0].maxVal = max(minMax[0].maxVal, uvec2(x, y));
			}
			
			if(len >= treshold)
			{
				useReprojection = 1;
				minMax[1].minVal = min(minMax[1].minVal, uvec2(x, y));
				minMax[1].maxVal = max(minMax[1].maxVal, uvec2(x, y));
			}
		}
	}
	
	atomicMin(shExtents[0].minVal.x, minMax[0].minVal.x);
	atomicMin(shExtents[0].minVal.y, minMax[0].minVal.y);
	atomicMax(shExtents[0].maxVal.x, minMax[0].maxVal.x);
	atomicMax(shExtents[0].maxVal.y, minMax[0].maxVal.y);
														
	atomicMin(shExtents[1].minVal.x, minMax[1].minVal.x);
	atomicMin(shExtents[1].minVal.y, minMax[1].minVal.y);
	atomicMax(shExtents[1].maxVal.x, minMax[1].maxVal.x);
	atomicMax(shExtents[1].maxVal.y, minMax[1].maxVal.y);
	
	if(useReprojection!=0)
	{
		atomicExchange(shUseReprojection, 1);
	}

	barrier();
	
	//Compute matrices
	if(lIndex==0)
	{
		const bool areData = shExtents[0].minVal.x != (heatmapResolution.x -1);
		
		if(areData)
		{
			lightP[2*wgid + 0] = GetProjectionMatrix(0);
			lightP[2*wgid + 1] = GetProjectionMatrix(1);
		}
		nofMatrices[wgid] = areData ? (shUseReprojection + 1) : 0;
	}
}
).";

	const char* izbFillCS = R".(

layout(local_size_x = WG_SIZE) in;

layout (binding = 0, r32i)  uniform coherent iimage2DArray head;
layout (binding = 1, r32i)  uniform coherent iimage2D list;
layout (binding = 2, r32ui) uniform coherent uimage2DArray maxDepth;


layout(packed, binding = 0) uniform matData
{
	mat4 lightP[12];
	uint nofMatrices[6];
};

uniform uvec2 screenResolution;
uniform uvec2 lightResolution;
uniform vec3 lightPos;
uniform mat4 lightV[6];

layout(binding=0) uniform sampler2D posTexture;
layout(binding=1) uniform sampler2D normalTexture;

bool isInFrustum(vec4 lightFragPos)
{
	const float w = lightFragPos.w;
	return (lightFragPos.x <= w && lightFragPos.x >= -w && lightFragPos.y <= w && lightFragPos.y >=-w && lightFragPos.z <= w && lightFragPos.z >= -w);
}

void main()
{
	const uint gid = gl_GlobalInvocationID.x;
	
	if(gid >= (screenResolution.x * screenResolution.y))
    {
        return;
    }
	
	const ivec2 coords = ivec2(gid % screenResolution.x, gid / screenResolution.x);
	
	//Read the position sample
	vec4 worldFragPos = texelFetch(posTexture, coords, 0);
	vec3 worldNormal = texelFetch(normalTexture, coords, 0).xyz;
	
	const float NdL = dot(worldNormal, normalize(lightPos - worldFragPos.xyz));
	
	// Early N dot L reject - this fragment is shadowed by lighting implicitly
	if(worldFragPos.w == 0.f || NdL < 0)
	{
		return;
	}
	
	for(int f = 0; f<6; ++f)
	{
		if(nofMatrices[f]==0)
		{
			continue;
		}
		
		//Project to light-space
		const mat4 lightVP0 = lightP[2*f + 0] * lightV[f];
		const vec4 lightFragPos0 = lightVP0 * worldFragPos;
		
		const mat4 lightVP1 = lightP[2*f + 1] * lightV[f];
		const vec4 lightFragPos1 = lightVP1 * worldFragPos; 
		
		const bool isIn0 = isInFrustum(lightFragPos0);
		const bool isIn1 = isInFrustum(lightFragPos1) && nofMatrices[f]==2;
		
		vec4 lightFragPos = isIn1 ? lightFragPos1 : lightFragPos0;
		int arrayIndex = 2*f + int(isIn1);
		
		if(isIn0) //generally, inside light's frustum
		{
			//... to texture space
			lightFragPos /= lightFragPos.w;
			lightFragPos.xyz = 0.5f * lightFragPos.xyz + 0.5f;
			
			//... to light space coords
			const ivec3 lightSpaceCoords = ivec3(lightFragPos.xy * lightResolution, arrayIndex);
			const int pos = int(coords.x + screenResolution.x * coords.y);

			const int previousHead = imageAtomicExchange(head, lightSpaceCoords, pos);
			imageAtomicMax(maxDepth, lightSpaceCoords, floatBitsToUint(lightFragPos.z));

			//Store previous ptr to current head
			imageStore(list, coords, ivec4(previousHead, 0, 0, 0));
			memoryBarrier();
		}
	}
}
).";


//ZBuffer optimization
const char* vsZFill = R".(
#version 450 core

void main()
{
	gl_Position = vec4(-1+2*(gl_VertexID/2),-1+2*(gl_VertexID%2),0,1);
}
).";


const char* gsZFill = R".(
#version 450 core

layout(triangles, invocations = 6) in;
layout(triangle_strip, max_vertices=6) out;

layout(packed, binding = 0) uniform matData
{
	uint nofMatrices[6];
};

void main()
{
	for(int i=0; i<nofMatrices[gl_InvocationID]; i++)
	{
		const int layer = 2*gl_InvocationID + i;
		
		gl_Layer = layer;
		gl_Position = gl_in[0].gl_Position; 
		EmitVertex();

		gl_Layer = layer;
		gl_Position = gl_in[1].gl_Position; 
		EmitVertex();

		gl_Layer = layer;
		gl_Position = gl_in[2].gl_Position; 
		EmitVertex();

		EndPrimitive();
	}
}
).";

const char* fsZFill = R".(
#version 450 core

layout(binding=0) uniform sampler2DArray maxDepth;

void main()
{
	const ivec3 coords = ivec3(gl_FragCoord.xy, gl_Layer);
	float depth = texelFetch(maxDepth, coords, 0).x;

	gl_FragDepth = depth;
}
).";


// Shadow mask
const char* vsShadowMask = R".(
#version 450 core

in vec3 position;

void main()
{
	gl_Position = vec4(position, 1);
}
).";

const char* gsShadowMask = R".(
#version 450 core

layout(triangles, invocations = 6) in;
layout(triangle_strip, max_vertices=6) out;

uniform vec3 lightPos;
uniform float bias = 0.001f;
uniform mat4 lightV[6];


layout(packed, binding = 0) uniform matData
{
	mat4 lightP[12];
	uint nofMatrices[6];
};

out flat vec4 plane0;
out flat vec4 plane1;
out flat vec4 plane2;
out flat vec4 plane3;

// Same stuff as in DPM
int greaterVec(vec3 a,vec3 b)
{
	return int(dot(ivec3(sign(a-b)),ivec3(4,2,1)));
}

vec4 getPlane(vec3 A,vec3 B,vec3 L)
{
	if(greaterVec(A,B)>0)
    {
		vec3 n=normalize(cross(A-B,L-B));
		return vec4(-n,dot(n,B));
	}
    else
    {
		vec3 n=normalize(cross(B-A,L-A));
		return vec4(n,-dot(n,A));
	}
}

vec4 getPlaneTri(vec3 A,vec3 B,vec3 C)
{
	vec3 n=normalize(cross(B-A,C-A));
	return vec4(n,-dot(n,A));
}

void main()
{	
	//Calculate shadow planes
	vec3 v0 = gl_in[0].gl_Position.xyz;
	vec3 v1 = gl_in[1].gl_Position.xyz;
	vec3 v2 = gl_in[2].gl_Position.xyz;
	
	vec4 e0 = getPlane(v0,v1,lightPos);
	vec4 e1 = getPlane(v1,v2,lightPos);
	vec4 e2 = getPlane(v2,v0,lightPos);
	vec4 e3 = getPlaneTri(
        v0 + bias*normalize(v0-lightPos.xyz),
        v1 + bias*normalize(v1-lightPos.xyz),
        v2 + bias*normalize(v2-lightPos.xyz));
	
	
	if(dot(e3, vec4(lightPos, 1))<0)
    {
		e0 *= -1;
		e1 *= -1;
		e2 *= -1;
		e3 *= -1;
	}
	
	for(int m = 0; m < nofMatrices[gl_InvocationID]; ++m)
	{
		const int layer = 2*gl_InvocationID + m;
		const mat4 lightVP = lightP[layer] * lightV[gl_InvocationID];
		
		//Output vertices
		plane0 = e0;
		plane1 = e1;
		plane2 = e2;
		plane3 = e3;
			
		gl_Layer = layer;
		gl_Position = lightVP * gl_in[0].gl_Position;
		EmitVertex();
			
		plane0 = e0;
		plane1 = e1;
		plane2 = e2;
		plane3 = e3;
			
		gl_Layer = layer;
		gl_Position = lightVP * gl_in[1].gl_Position;
		EmitVertex();
	
		plane0 = e0;
		plane1 = e1;
		plane2 = e2;
		plane3 = e3;
			
		gl_Layer = layer;
		gl_Position = lightVP * gl_in[2].gl_Position;
		EmitVertex();
	
		EndPrimitive();
	}
}
).";

const char* fsShadowMask = R".(
#version 450 core

layout(binding=0, r32i) uniform iimage2DArray head; //in light space
layout(binding=1, r32i) uniform iimage2D list; //in screen space
layout(binding=2, r32f) uniform writeonly image2D shadowMask; //in screen space

layout(binding=0) uniform sampler2D  positions; //in screen space

in flat vec4 plane0;
in flat vec4 plane1;
in flat vec4 plane2;
in flat vec4 plane3;

uniform uvec2 screenResolution;

bool isPointInside(vec4 v, vec4 p1, vec4 p2, vec4 p3, vec4 p4)
{
    return dot(v, p1)<=0 && dot(v, p2)<=0 && dot(v, p3)<=0 && dot(v, p4)<=0;
}

void main()
{
	//Acquire head
	const ivec3 lighSpaceCoords = ivec3(gl_FragCoord.xy, gl_Layer);
    int currentFragPos = imageLoad(head, lighSpaceCoords).x;//texelFetch(head, lighSpaceCoords, 0).x;
	int previousFragPos = -1;
	ivec2 previousScreenCoords = ivec2(-1, -1);	

	while(currentFragPos >=0)
	{
		const ivec2 screenSpaceCoords = ivec2(currentFragPos % screenResolution.x, currentFragPos / screenResolution.x);
		const vec3 pos = texelFetch(positions, screenSpaceCoords, 0).xyz;
		const bool isInsideFrustum = isPointInside(vec4(pos, 1), plane0, plane1, plane2, plane3);
		
		const int nextFragPos = imageLoad(list, screenSpaceCoords).x;//texelFetch(list, screenSpaceCoords, 0).x;

        if(isInsideFrustum)
        {
            imageStore(shadowMask, screenSpaceCoords, vec4(0.f, 0, 0, 1));

			//Optimization - remove the node from the list
			//Store nextPtr to parent
			if(previousFragPos < 0)
			{
				imageStore(head, lighSpaceCoords, ivec4(nextFragPos, 0, 0, 0));
			}
			else
			{
				imageStore(list, previousScreenCoords, ivec4(nextFragPos, 0, 0, 0));
			}
        }
		
		//Get next pos
		previousFragPos = currentFragPos;
		previousScreenCoords = screenSpaceCoords;
		currentFragPos = nextFragPos;
	}
}
).";
};

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetHeatmapCS(uint32_t wgSize)
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_OmniHeatmapShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/

	return std::make_shared<Shader>(GL_COMPUTE_SHADER,
		"#version 450 core\n",
		Shader::define("WG_SIZE", wgSize),
		Shader::define("SHADOW_OUTSIDE_FRUSTUM"),
		OFTS::heatmapCS);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetMatrixCS()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_OmniExtentsShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/

	return std::make_shared<Shader>(GL_COMPUTE_SHADER, OFTS::matrixCS);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetIzbFillCS(uint32_t wgSize)
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_OmniFillShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/
	return std::make_shared<Shader>(GL_COMPUTE_SHADER,
		"#version 450 core\n",
		Shader::define("WG_SIZE", wgSize),
		OFTS::izbFillCS);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetZBufferFillVS()
{
	return std::make_shared<Shader>(GL_VERTEX_SHADER, OFTS::vsZFill);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetZBufferFillGS()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_OmniZbuffGS.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_GEOMETRY_SHADER, program);
	//*/
	return std::make_shared<Shader>(GL_GEOMETRY_SHADER, OFTS::gsZFill);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetZBufferFillFS()
{
	return std::make_shared<Shader>(GL_FRAGMENT_SHADER, OFTS::fsZFill);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetShadowMaskVS()
{
	return std::make_shared<Shader>(GL_VERTEX_SHADER, OFTS::vsShadowMask);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetShadowMaskGS()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_OmniShadowMaskGS.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_GEOMETRY_SHADER, program);
	//*/
	return std::make_shared<Shader>(GL_GEOMETRY_SHADER, OFTS::gsShadowMask);
}

std::shared_ptr<ge::gl::Shader> OftsShaderGen::GetShadowMaskFS()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_OmniShadowMaskFS.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/

	return std::make_shared<Shader>(GL_FRAGMENT_SHADER, OFTS::fsShadowMask);
}

