#include <Sintorn2/quantizeZShader.h>

std::string const sintorn2::quantizeZShader= R".(

#ifndef WINDOW_X
#define WINDOW_X 512
#endif//WINDOW_X

#ifndef WINDOW_Y
#define WINDOW_Y 512
#endif//WINDOW_Y

#ifndef TILE_X
#define TILE_X 8
#endif//TILE_X

#ifndef TILE_Y
#define TILE_Y 8
#endif//TILE_Y

#ifndef MIN_Z_BITS
#define MIN_Z_BITS 9
#endif//MIN_Z_BITS

#ifndef NEAR
#define NEAR 0.01f
#endif//NEAR

#ifndef FOVY
#define FOVY 1.5707963267948966f
#endif//FOVY


).";
