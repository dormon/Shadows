#include <FTS_shaderGen.h>

#include <geGL/Program.h>
#include <geGL/Shader.h>

#include <fstream>
#include <algorithm>

using namespace ge;
using namespace gl;

std::string fillCsSource = R".(
#define HEAD_UNLOCKED 0u
#define HEAD_LOCKED 1u

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
	const uint gid=gl_GlobalInvocationID.x;
	if(gid >= (screenResolution.x * screenResolution.y))
    {
        return;
    }
	
	const uvec2 coords = uvec2(gid % screenResolution.x, gid / screenResolution.x);
	const vec2 tcoords = vec2(coords.x / float(screenResolution.x), coords.y / float(screenResolution.y));
	
	//Read the position sample
	vec4 worldFragPos = texture(posTexture, tcoords);
	
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
		
		//... to light texture rsolution
		const vec2 texelSize = vec2(1.f/lightResolution.x, 1.f/lightResolution.y);
		const ivec2 lightSpaceCoords = ivec2((lightFragPos.xy + texelSize)* lightResolution);
		
		const int pos = int(coords.x + screenResolution.x * coords.y);

		const int previousHead = imageAtomicExchange(head, lightSpaceCoords, pos);
		memoryBarrier();

		if(previousHead != HEAD_POINTER_INVALID)
		{
			ivec2 previousCoords = ivec2(previousHead % screenResolution.x, previousHead / screenResolution.x);
			imageStore(list, previousCoords, ivec4(pos, 0, 0, 0));
		}
	}
}
).";

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetZbufferFillCS(uint32_t wgSize)
{
	//std::ifstream t1("C:\\Users\\Jofo\\Desktop\\FTS_FillShader.glsl");
	//std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	//*/

	return std::make_shared<Shader>(GL_COMPUTE_SHADER, 
		"#version 450 core\n",
		Shader::define("WG_SIZE", wgSize),
		fillCsSource);
}
