#pragma once

#include <memory>

namespace ge
{
	namespace gl
	{
		class Program;
		class Shader;
	}
}


class OftsShaderGen
{
public:
	std::shared_ptr<ge::gl::Shader> GetHeatmapCS(uint32_t wgSize);

	std::shared_ptr<ge::gl::Shader> GetMatrixCS();

	std::shared_ptr<ge::gl::Shader> GetIzbFillCS(uint32_t wgSize);

	std::shared_ptr<ge::gl::Shader> GetZBufferFillVS();
	std::shared_ptr<ge::gl::Shader> GetZBufferFillGS();
	std::shared_ptr<ge::gl::Shader> GetZBufferFillFS();

	std::shared_ptr<ge::gl::Shader> GetShadowMaskVS();
	std::shared_ptr<ge::gl::Shader> GetShadowMaskGS();
	std::shared_ptr<ge::gl::Shader> GetShadowMaskFS();
};