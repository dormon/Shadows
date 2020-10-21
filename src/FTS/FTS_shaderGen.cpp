#include <FTS_shaderGen.h>

#include <geGL/Program.h>
#include <geGL/Shader.h>

#include <fstream>
#include <algorithm>

using namespace ge;
using namespace gl;

std::string FtsShaderGen::GetZbufferFillProgramString()
{
	return "";
}

//ge::gl::Shader::define("FAR", ffar)

std::shared_ptr<ge::gl::Shader> FtsShaderGen::GetZbufferFillCS(uint32_t wgSize)
{
	
	std::ifstream t1("C:\\Users\\Jofo\\Desktop\\FTS_FillShader.glsl");
	std::string program((std::istreambuf_iterator<char>(t1)), std::istreambuf_iterator<char>());
	//*/

	//std::string program = GetZbufferFillProgramString();

	return std::make_shared<Shader>(GL_COMPUTE_SHADER, program);
}
