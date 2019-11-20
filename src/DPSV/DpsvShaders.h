#pragma once

#include <geGL/Shader.h>

#include <memory>
#include <vector>

std::shared_ptr<ge::gl::Shader> getDpsvBuildCS(unsigned int wgSiz, bool enableFrontFaceCulling, bool enableDepthOptim);

std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvStackProgramShaders(bool enableDepthOptim);
std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvStacklessProgramShaders(bool enableDepthOptim);
std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvHybridProgramShaders(bool enableDepthOptim);

//Helper function
std::shared_ptr<ge::gl::Shader> getDpsvVertexShader();