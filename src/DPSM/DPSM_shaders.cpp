#include "DPSM_shaders.h"

std::string getDpsmCreateVS()
{
	return
		R".(
#version 450 core

uniform mat4 lightV;

layout(location=0) in vec3 pos;

void main()
{
	gl_Position = lightV * vec4(pos, 1);
}
).";
}

std::string getDpsmCreateGS()
{
	return
		R".(
#version 450 core

layout(triangles)in;
layout(triangle_strip, max_vertices=6) out;

uniform float nearClip;
uniform float farClip;

out float depth;
out float clipDepth;
void main()
{
	for(int d = 0; d < 2; d++)
	{
		for(int v = 0; v < 3; v++)
		{
			float parDirection = (d==0) ? 1.0f : -1.0f;
			gl_Layer = d;
		
			vec4 pos = gl_in[v].gl_Position;
			pos /= pos.w;
			pos.z *= parDirection;

			float vLen = length(pos.xyz);
			pos /= vLen;
		
			clipDepth = pos.z;

			pos.x /= pos.z + 1.0f;
			pos.y /= pos.z + 1.0f;

			pos.z = (vLen - nearClip) / (farClip - nearClip);
			pos.w = 1.0f;

			depth = pos.z;

			gl_Position = pos;
			EmitVertex();
		}

		EndPrimitive();
	}
}
).";
}

std::string getDpsmCreateFS()
{
	return
		R".(
#version 450 core

in float depth;
in float clipDepth;

void main()
{
	if(clipDepth < 0)
	{
		discard;
	}

	gl_FragDepth = depth;
}
).";
}

std::string getDpsmFillVS()
{
	return
		R".(
#version 450 core

void main()
{
	gl_Position = vec4(-1+2*(gl_VertexID/2),-1+2*(gl_VertexID%2),0,1);
}
).";
}

std::string getDpsmFillFS()
{
	return
		R".(
#version 450 core

layout(location=0) out float fColor;

layout(binding=0) uniform sampler2D			position;
layout(binding=1) uniform sampler2DArrayShadow	shadowMap;

uniform mat4  lightV;
uniform float nearClip;
uniform float farClip;

void main()
{
	ivec2 Coord = ivec2(gl_FragCoord.xy);
	vec3 viewSamplePosition = texelFetch(position,Coord,0).xyz;
	vec4 lPos = lightV * vec4(viewSamplePosition, 1);
	
    float lLen = length(lPos.xyz);
	lPos /= lLen;

	float SMDepth = 0.0f;
	vec2 texCoords;
	
	const float shadowBias = 0.0005f;
	const float sceneDepth = (lLen - nearClip) / (farClip - nearClip);

	//Sampling SM
	if(lPos.z >=0)
	{
		texCoords.x = (lPos.x /  (1.0f + lPos.z)) * 0.5f + 0.5f; 
		texCoords.y = (lPos.y /  (1.0f + lPos.z)) * 0.5f + 0.5f; 

		SMDepth = texture(shadowMap, vec4(texCoords, 0, sceneDepth - shadowBias)).x;
	}
	else
	{
		texCoords.x = (lPos.x /  (1.0f - lPos.z)) * 0.5f + 0.5f; 
		texCoords.y = (lPos.y /  (1.0f - lPos.z)) * 0.5f + 0.5f; 

		SMDepth = texture(shadowMap, vec4(texCoords, 1, sceneDepth - shadowBias)).x;
	}
	
	fColor = SMDepth;
}
).";
}
