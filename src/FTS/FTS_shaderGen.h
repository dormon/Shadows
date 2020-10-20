#pragma once

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace ge
{
	namespace gl
	{
		class Program;
		class Shader;
	}
}

class FtsShaderGen
{
public:

	std::shared_ptr<ge::gl::Shader> GetZbufferFillCS(uint32_t wgSize);

private:

	std::string GetZbufferFillProgramString();
	
};