#pragma once

#include <geGL/Shader.h>

#include <memory>
#include <vector>

std::shared_ptr<ge::gl::Shader> getDpsvBuildCS(unsigned int wgSiz);

std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvStackProgramShaders();
std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvStacklessProgramShaders();
std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvHybridProgramShaders();

//Helper function
std::shared_ptr<ge::gl::Shader> getDpsvVertexShader();