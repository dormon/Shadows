#include <divRoundUp.h>

std::size_t divRoundUp(std::size_t a, std::size_t b) {
  return (a / b) + ((a % b) > 0 ? 1 : 0);
}
