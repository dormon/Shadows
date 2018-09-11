#include <util.h>
#include <cstring>
#include <sstream>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>

uint32_t getDispatchSize(std::size_t n, std::size_t wgs) {
  return static_cast<uint32_t>((n / wgs) + ((n % wgs) > 0 ? 1 : 0));
}

std::size_t divRoundUp(std::size_t a, std::size_t b) {
  return (a / b) + ((a % b) > 0 ? 1 : 0);
}

glm::vec2 vector2vec2(std::vector<float> const& v) {
  assert(v.size() >= 2);
  return glm::vec2(v[0], v[1]);
}
glm::vec3 vector2vec3(std::vector<float> const& v) {
  assert(v.size() >= 3);
  return glm::vec3(v[0], v[1], v[2]);
}
glm::vec4 vector2vec4(std::vector<float> const& v) {
  assert(v.size() >= 4);
  return glm::vec4(v[0], v[1], v[2], v[3]);
}
glm::ivec2 vector2ivec2(std::vector<int32_t> const& v) {
  assert(v.size() >= 2);
  return glm::ivec2(v[0], v[1]);
}
glm::ivec3 vector2ivec3(std::vector<int32_t> const& v) {
  assert(v.size() >= 3);
  return glm::ivec3(v[0], v[1], v[2]);
}
glm::ivec4 vector2ivec4(std::vector<int32_t> const& v) {
  assert(v.size() >= 4);
  return glm::ivec4(v[0], v[1], v[2], v[3]);
}
glm::uvec2 vector2uvec2(std::vector<uint32_t> const& v) {
  assert(v.size() >= 2);
  return glm::uvec2(v[0], v[1]);
}
glm::uvec3 vector2uvec3(std::vector<uint32_t> const& v) {
  assert(v.size() >= 3);
  return glm::uvec3(v[0], v[1], v[2]);
}
glm::uvec4 vector2uvec4(std::vector<uint32_t> const& v) {
  assert(v.size() >= 4);
  return glm::uvec4(v[0], v[1], v[2], v[3]);
}
glm::vec2 vector2vec2(std::vector<double> const& v) {
  assert(v.size() >= 2);
  return glm::vec2(v[0], v[1]);
}
glm::vec3 vector2vec3(std::vector<double> const& v) {
  assert(v.size() >= 3);
  return glm::vec3(v[0], v[1], v[2]);
}
glm::vec4 vector2vec4(std::vector<double> const& v) {
  assert(v.size() >= 4);
  return glm::vec4(v[0], v[1], v[2], v[3]);
}

size_t getWavefrontSize(size_t w) {
  if (w != 0) return w;
  std::string renderer = std::string((char*)ge::gl::glGetString(GL_RENDERER));
  std::string vendor   = std::string((char*)ge::gl::glGetString(GL_VENDOR));
  std::cout << renderer << std::endl;
  std::cout << vendor << std::endl;
  if (vendor.find("AMD") != std::string::npos ||
      renderer.find("AMD") != std::string::npos)
    return 64;
  else if (vendor.find("NVIDIA") != std::string::npos ||
           renderer.find("NVIDIA") != std::string::npos)
    return 32;
  else {
    std::cerr << "WARNING: renderer is not NVIDIA or AMD, setting "
                 "wavefrontSize to 32"
              << std::endl;
    return 32;
  }
}

std::string uvec2ToStr(glm::uvec2 const& v)
{
  std::stringstream ss;
  ss << v.x << "," << v.y;
  return ss.str();
}

size_t align(size_t what,size_t alignment){
  return (what / alignment) * alignment + (size_t)((what % alignment)!=0)*alignment;
}
