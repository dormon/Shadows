#include <FTS_shaderGen.h>

#include <geGL/Program.h>
#include <geGL/Shader.h>

#include <fstream>
#include <algorithm>

using namespace ge;
using namespace gl;

std::string const heatmapShaderSource = R".(

layout(local_size_x = WG_SIZE) in;

layout(binding = 0, r32ui) uniform coherent uimage2D heatMap;
layout(binding = 1, r32f)  uniform writeonly image2D shadowMask;

layout(binding=0) uniform sampler2D posTexture;
layout(binding=1) uniform sampler2D normalTexture;

uniform uvec2 screenResolution;
uniform uvec2 lightResolution;
uniform vec3  lightPos;
uniform mat4  lightVP;

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
	
	vec4 lightProjPos = lightVP * worldFragPos;
	
	// If inside light frustum, add the fragment to the list
	const float w = lightProjPos.w;
	const bool isInsideFrustum = lightProjPos.x <= w && lightProjPos.x >= -w && lightProjPos.y <= w && lightProjPos.y >=-w && lightProjPos.z <= w && lightProjPos.z >= -w;
	const bool isValidVisible = hasValidSample && isInsideFrustum;
	
	//... to texture space
	lightProjPos /= w;
	lightProjPos.xyz = 0.5f * lightProjPos.xyz + 0.5f;
	
	//... to light space coords
	const ivec2 lightSpaceCoords = ivec2(lightProjPos.xy * lightResolution);
	
	//Add to heat map if valid
	if(isValidVisible)
    {
		imageAtomicAdd(heatMap, lightSpaceCoords, 1u);
	}
	
	if(hasValidSample && (!isInsideFrustum))
	{
		imageStore(shadowMask, coords, vec4(0));
	}
}
).";

std::string const matrixSahderSource = R".(
#version 450 core

layout(local_size_x = 32, local_size_y = 32) in;

layout(binding = 0, r32ui) uniform readonly uimage2D heatMap;

layout(std430, binding = 0) restrict writeonly buffer _proj{ mat4 projMatrix[1]; };

struct S_MinMax
{
	uvec2 minVal;
	uvec2 maxVal;
};

shared S_MinMax shExtents;

uniform uvec2 lightResolution;
uniform vec4 frustumParams;


mat4 GetProjectionMatrix()
{
	const float dw = frustumParams.x / float(lightResolution.x);
	const float dh = frustumParams.y / float(lightResolution.y);
	
	const uint minX = shExtents.minVal.x;
	const uint minY = shExtents.minVal.y;
	const uint maxX = shExtents.maxVal.x;
	const uint maxY = shExtents.maxVal.y;
	
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

//Runs only as a single workgroup
void main()
{
	const uint gid = gl_GlobalInvocationID.x;
	
	if(gid==0)
	{
		shExtents.minVal = lightResolution - uvec2(1, 1);
		shExtents.maxVal = uvec2(0, 0);
	}
	
	const uvec2 blockSize = lightResolution / 32;
	
	const uvec2 start = gl_GlobalInvocationID.xy * blockSize;
	const uvec2 end = start + blockSize;
	
	S_MinMax minMax;
	minMax.minVal = uvec2(lightResolution - uvec2(1, 1));
	minMax.maxVal = uvec2(0, 0);
	
	for(uint x = start.x; x < end.x; ++x)
	{
		for(uint y = start.y; y < end.y; ++y)
		{
			const uint len = imageLoad(heatMap, ivec2(x, y)).x;
			if(len > 0)
			{
				minMax.minVal = min(minMax.minVal, uvec2(x, y));
				minMax.maxVal = max(minMax.maxVal, uvec2(x, y));
			}
		}
	}
	
	memoryBarrier();
	
	atomicMin(shExtents.minVal.x, minMax.minVal.x);
	atomicMin(shExtents.minVal.y, minMax.minVal.y);
	atomicMax(shExtents.maxVal.x, minMax.maxVal.x);
	atomicMax(shExtents.maxVal.y, minMax.maxVal.y);
	
	memoryBarrier();
	
	//Compute matrix
	if(gid==0)
	{
		projMatrix[0] = GetProjectionMatrix();
	}
}
).";

// IZB Fill
std::string fillCsSource = R".(

#define HEAD_POINTER_INVALID (-1)

layout(local_size_x = WG_SIZE) in;

layout (binding = 0, r32i) uniform coherent iimage2D head;
layout (binding = 1, r32i) uniform coherent iimage2D list;
layout (binding = 2, r32ui) uniform coherent uimage2D maxDepth;

layout(std140, binding = 0) uniform matData
{
	mat4 lightP;
};

uniform uvec2 screenResolution;
uniform uvec2 lightResolution;
uniform vec3 lightPos;
uniform mat4 lightV;

layout(binding=0) uniform sampler2D posTexture;
layout(binding=1) uniform sampler2D normalTexture;

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
	
	//Project to light-space
	mat4 lightVP = lightP * lightV;
	vec4 lightFragPos = lightVP * worldFragPos;
	
	// If inside light frustum, add the fragment to the list
	const float w = lightFragPos.w;
	if(lightFragPos.x <= w && lightFragPos.x >= -w && lightFragPos.y <= w && lightFragPos.y >=-w && lightFragPos.z <= w && lightFragPos.z >= -w)
    {
		//... to texture space
	    lightFragPos /= lightFragPos.w;
		lightFragPos.xyz = 0.5f * lightFragPos.xyz + 0.5f;
		
		//... to light space coords
		const ivec2 lightSpaceCoords = ivec2(lightFragPos.xy * lightResolution);
		const int pos = int(coords.x + screenResolution.x * coords.y);

		const int previousHead = imageAtomicExchange(head, lightSpaceCoords, pos);
		imageAtomicMax(maxDepth, lightSpaceCoords, floatBitsToUint(lightFragPos.z));
		//imageAtomicAdd(counter, lightSpaceCoords, 1u);
		memoryBarrier();

		//Store previous ptr to current head
		imageStore(list, coords, ivec4(previousHead, 0, 0, 0));
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

const char* fsZFill = R".(
#version 450 core

layout(binding=0) uniform sampler2D maxDepth;

void main()
{
	const ivec2 coords = ivec2(gl_FragCoord.xy);
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

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

uniform vec3 lightPos;
uniform float bias = 0.001f;
uniform mat4 lightV;

layout(std140, binding = 0) uniform matData
{
	mat4 lightP;
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
	
	const mat4 lightVP = lightP * lightV;
	
	if(dot(e3, vec4(lightPos, 1))<0)
    {
		e0 *= -1;
		e1 *= -1;
		e2 *= -1;
		e3 *= -1;
	}

	//Output vertices
	for(int i=0; i<3; i++)
	{
		plane0 = e0;
		plane1 = e1;
		plane2 = e2;
		plane3 = e3;
		
		gl_Position = lightVP * gl_in[i].gl_Position;
		EmitVertex();
	}
	
	EndPrimitive();
}
).";

const char* fsShadowMask = R".(
#version 450 core

layout(binding=0, r32i) uniform iimage2D head; //in light space
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
	const ivec2 lighSpaceCoords = ivec2(gl_FragCoord.xy);
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

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetHeatmapCS(uint32_t wgSize)
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_HeatmapShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/

	
	return std::make_shared<Shader>(GL_COMPUTE_SHADER,
		"#version 450 core\n",
		Shader::define("WG_SIZE", wgSize),
		heatmapShaderSource
		);
	//*/
}

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetMatrixCS()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_ExtentsShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/
	
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, matrixSahderSource);
}


std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetIzbFillCS(uint32_t wgSize)
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_FillShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/

	
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, 
		"#version 450 core\n",
		Shader::define("WG_SIZE", wgSize),
		fillCsSource);
	//*/
}

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetZBufferFillVS()
{
	return std::make_shared<Shader>(GL_VERTEX_SHADER, vsZFill);
}

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetZBufferFillFS()
{
	return std::make_shared<Shader>(GL_FRAGMENT_SHADER, fsZFill);
}

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetShadowMaskVS()
{
	return std::make_shared<Shader>(GL_VERTEX_SHADER, vsShadowMask);
}

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetShadowMaskGS()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_GeomShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_GEOMETRY_SHADER, program);
	//*/
	return std::make_shared<Shader>(GL_GEOMETRY_SHADER, gsShadowMask);
}

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetShadowMaskFS()
{
	/*
	std::ifstream t1("C:\\Users\\ikobrtek\\Desktop\\FTS_FragShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_FRAGMENT_SHADER, program);
	//*/
	return std::make_shared<Shader>(GL_FRAGMENT_SHADER, fsShadowMask);
}
