#include <FTS_shaderGen.h>

#include <geGL/Program.h>
#include <geGL/Shader.h>

#include <fstream>
#include <algorithm>

using namespace ge;
using namespace gl;

std::string fillCsSource = R".(

#define HEAD_POINTER_INVALID (-1)

layout(local_size_x = WG_SIZE) in;

layout (binding = 0, r32i)  uniform coherent iimage2D head;
layout (binding = 1, r32i)  uniform coherent iimage2D list;

uniform uvec2 screenResolution;
uniform uvec2 lightResolution;
uniform mat4  lightVP;

uniform sampler2D posTexture;

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
	
	if(worldFragPos.w == 0.f)
	{
		return;
	}
	
	//Project to light-space
	vec4 lightFragPos = lightVP * worldFragPos;
	
	// If inside light frustum, add the fragment to the list
	const float w = lightFragPos.w;
	if(lightFragPos.x <= w && lightFragPos.x >= -w && lightFragPos.y <= w && lightFragPos.y >=-w && lightFragPos.z <= w && lightFragPos.z >= -w)
    {
		//... to texture space
	    lightFragPos /= lightFragPos.w;
		lightFragPos.xyz = 0.5f * lightFragPos.xyz + 0.5f;
		
		//... to light space coords
		const ivec2 lightSpaceCoords = ivec2(lightFragPos.xy* lightResolution);
		
		const int pos = int(coords.x + screenResolution.x * coords.y);

		const int previousHead = imageAtomicExchange(head, lightSpaceCoords, pos);
		memoryBarrier();

		//Store previous ptr to current head
		imageStore(list, coords, ivec4(previousHead, 0, 0, 0));
	}
}
).";

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

uniform mat4 lightVP;
uniform vec3 lightPos;
uniform float bias = 0.001f;

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

layout(binding=0) uniform isampler2D head; //in light space
layout(binding=1) uniform isampler2D list; //in scren space
layout(binding=2) uniform sampler2D  positions; //in screen space

layout (binding = 0, r32f) uniform writeonly image2D shadowMask; //in screen space

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
    int currentFragPos = texelFetch(head, lighSpaceCoords, 0).x;
	
	while(currentFragPos >=0)
	{
		const ivec2 screenSpaceCoords = ivec2(currentFragPos % screenResolution.x, currentFragPos / screenResolution.x);
		const vec3 pos = texelFetch(positions, screenSpaceCoords, 0).xyz;
		const bool isInsideFrustum = isPointInside(vec4(pos, 1), plane0, plane1, plane2, plane3);

        if(isInsideFrustum)
        {
            imageStore(shadowMask, screenSpaceCoords, vec4(0.f, 0, 0, 1));
        }
		
		//Get next pos
		currentFragPos = texelFetch(list, screenSpaceCoords, 0).x;
	}
}
).";

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetZbufferFillCS(uint32_t wgSize)
{
	/*
	std::ifstream t1("C:\\Users\\Jofo\\Desktop\\FTS_FillShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
	//*/


	return std::make_shared<Shader>(GL_COMPUTE_SHADER, 
		"#version 450 core\n",
		Shader::define("WG_SIZE", wgSize),
		fillCsSource);
	//*/
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
