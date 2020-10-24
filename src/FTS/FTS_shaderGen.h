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

	std::shared_ptr<ge::gl::Shader> GetShadowMaskVS();
	std::shared_ptr<ge::gl::Shader> GetShadowMaskGS();
	std::shared_ptr<ge::gl::Shader> GetShadowMaskFS();

private:
	
};