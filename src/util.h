#pragma once

#include<cstdint>
#include<vector>
#include<iostream>
#include<glm/glm.hpp>

uint32_t getDispatchSize(std::size_t n,std::size_t wgs);
std::size_t divRoundUp(std::size_t a,std::size_t b);
glm::vec2 vector2vec2(std::vector<float> const& v);
glm::vec3 vector2vec3(std::vector<float> const& v);
glm::vec4 vector2vec4(std::vector<float> const& v);
glm::ivec2 vector2ivec2(std::vector<int32_t> const& v);
glm::ivec3 vector2ivec3(std::vector<int32_t> const& v);
glm::ivec4 vector2ivec4(std::vector<int32_t> const& v);
glm::uvec2 vector2uvec2(std::vector<uint32_t> const& v);
glm::uvec3 vector2uvec3(std::vector<uint32_t> const& v);
glm::uvec4 vector2uvec4(std::vector<uint32_t> const& v);
glm::vec2 vector2vec2(std::vector<double> const& v);
glm::vec3 vector2vec3(std::vector<double> const& v);
glm::vec4 vector2vec4(std::vector<double> const& v);
std::string uvec2ToStr(glm::uvec2 const& v);

size_t getWavefrontSize(size_t w=0);
