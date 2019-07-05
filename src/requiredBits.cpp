#include <requiredBits.h>
#include <glm/glm.hpp>

size_t requiredBits(size_t number){
  return static_cast<size_t>(glm::ceil(glm::log2(static_cast<float>(number))));
}
