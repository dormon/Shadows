#include <RSSV/mortonShader.h>

const std::string rssv::mortonShader = R".(

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

// m - length of 3 bits together
// n - length of 2 bits together
// o - length of 1 bit  alone
//
//                      bits1Count|bits2shifts|1bit_offset|2bit_offset
// ..*|   .**|.**|.**|       o    |     n-1   |   2n+3m   |  2i+2+3m
// .*.|   .**|.**|.**|       o    |     n-1   |   2n+3m   |  2i+2+3m
// ...|   .**|.**|.**|       o    |     n-1   |   2n+3m   |  2i+2+3m
// ###################################################################
// ..*|   *.*|*.*|*.*|       o-1  |     n     |   2n+3m+1 |  2i+1+3m
// *..|   *.*|*.*|*.*|       o+1  |     n-1   |   2n+3m-1 |  2i+1+3m
// ...|   *.*|*.*|*.*|       o    |     n     |   2n+3m   |  2i+1+3m
// ###################################################################
// .*.|   **.|**.|**.|       o    |     n     |   2n+3m   |  2i+0+3m
// *..|   **.|**.|**.|       o    |     n     |   2n+3m   |  2i+0+3m
// ...|   **.|**.|**.|       o    |     n     |   2n+3m   |  2i+0+3m
// ###################################################################
// ..*|   ...|...|...|       o    |     n     |   2n+3m   |  xxxxxxx
// .*.|   ...|...|...|       o    |     n     |   2n+3m   |  xxxxxxx
// ...|   ...|...|...|       o    |     n     |   2n+3m   |  xxxxxxx

#line 46
uint morton2(uvec3 v){
  uint res = 0;
  uint counters[3] = {0,0,0};
  const uint limits[3] = {xBits,yBits,zBits};
  const uint allBits = xBits + yBits + zBits;
  uint a = 0;
  for(uint b=0;b<allBits;++b){
    res |= ((v[a]>>counters[a])&1u) << b;
    counters[a]++;
    a = (a+1u)%3u;
    if(counters[a] >= limits[a])a = (a+1u)%3u;
    if(counters[a] >= limits[a])a = (a+1u)%3u;
  }
  return res;
}

uint morton(uvec3 v){
  const uint shortestZ     = uint(shortestAxis == 2u);
  const uint shortestY     = uint(shortestAxis == 1u);
  const uint isMiddle      = uint(bitTogether[1] > 0u);
  const uint isLongest     = uint(bitTogether[2] > 0u);
  const uint longestZ      = uint(longestAxis == 2u) * isLongest;
  const uint longestX      = uint(longestAxis == 0u) * isLongest;

  const uint bits2Shifts   = uint(uint(bitTogether[1] - uint(shortestZ | (shortestY & longestZ))) * isMiddle);

  const uint bits2OffsetB   = bitTogether[0]*3u + shortestAxis;
  const uint bits2Offset00  = bits2OffsetB + 2* 0;
  const uint bits2Offset01  = bits2OffsetB + 2* 1;
  const uint bits2Offset02  = bits2OffsetB + 2* 2;
  const uint bits2Offset03  = bits2OffsetB + 2* 3;
  const uint bits2Offset04  = bits2OffsetB + 2* 4;
  const uint bits2Offset05  = bits2OffsetB + 2* 5;
  const uint bits2Offset06  = bits2OffsetB + 2* 6;
  const uint bits2Offset07  = bits2OffsetB + 2* 7;
  const uint bits2Offset08  = bits2OffsetB + 2* 8;
  const uint bits2Offset09  = bits2OffsetB + 2* 9;
  const uint bits2Offset10  = bits2OffsetB + 2*10;

  const uint bits2LMask00 = uint((1u << bits2Offset00)-1u);
  const uint bits2LMask01 = uint((1u << bits2Offset01)-1u);
  const uint bits2LMask02 = uint((1u << bits2Offset02)-1u);
  const uint bits2LMask03 = uint((1u << bits2Offset03)-1u);
  const uint bits2LMask04 = uint((1u << bits2Offset04)-1u);
  const uint bits2LMask05 = uint((1u << bits2Offset05)-1u);
  const uint bits2LMask06 = uint((1u << bits2Offset06)-1u);
  const uint bits2LMask07 = uint((1u << bits2Offset07)-1u);
  const uint bits2LMask08 = uint((1u << bits2Offset08)-1u);
  const uint bits2LMask09 = uint((1u << bits2Offset09)-1u);
  const uint bits2LMask10 = uint((1u << bits2Offset10)-1u);

  const uint bits2HMask00 = (~bits2LMask00)<<1u;
  const uint bits2HMask01 = (~bits2LMask01)<<1u;
  const uint bits2HMask02 = (~bits2LMask02)<<1u;
  const uint bits2HMask03 = (~bits2LMask03)<<1u;
  const uint bits2HMask04 = (~bits2LMask04)<<1u;
  const uint bits2HMask05 = (~bits2LMask05)<<1u;
  const uint bits2HMask06 = (~bits2LMask06)<<1u;
  const uint bits2HMask07 = (~bits2LMask07)<<1u;
  const uint bits2HMask08 = (~bits2LMask08)<<1u;
  const uint bits2HMask09 = (~bits2LMask09)<<1u;
  const uint bits2HMask10 = (~bits2LMask10)<<1u;

  const uint bits1Count    = uint(bitTogether[2] - uint(shortestY & longestX) + uint(shortestY & longestZ)) * isLongest;
  const uint bits1used     = bitLength[2] - bits1Count;
  const uint bits1DstMask  = uint((1u<<(bitTogether[0]*3u + bitTogether[1]*2u + uint(shortestY & longestX) - uint(longestZ & shortestY))) -1u);
  const uint bits1SrcShift = bitTogether[0]*3u + bitTogether[1]*2u - uint(shortestY & longestZ) + uint(shortestY & longestX)  - bits1used;
  const uint bits1SrcMask  = ~((1u<<bits1used)-1u);

  uint res = 0;
  uint vv;
  vv   = (v[0] * (0x00010001u<<0u)) & (0xFF0000FFu<<0u);
  vv   = (vv   * (0x00000101u<<0u)) & (0x0F00F00Fu<<0u);
  vv   = (vv   * (0x00000011u<<0u)) & (0xC30C30C3u<<0u);
  res |= (vv   * (0x00000005u<<0u)) & (0x49249249u<<0u);

  vv   = (v[1] * (0x00010001u<<0u)) & (0xFF0000FFu<<0u);
  vv   = (vv   * (0x00000101u<<0u)) & (0x0F00F00Fu<<0u);
  vv   = (vv   * (0x00000011u<<0u)) & (0xC30C30C3u<<0u);
  res |= (vv   * (0x00000005u<<1u)) & (0x49249249u<<1u);

  vv   = (v[2] * (0x00010001u<<0u)) & (0xFF0000FFu<<0u);
  vv   = (vv   * (0x00000101u<<0u)) & (0x0F00F00Fu<<0u);
  vv   = (vv   * (0x00000011u<<0u)) & (0xC30C30C3u<<0u);
  res |= (vv   * (0x00000005u<<2u)) & (0x49249249u<<2u);

  if(0  < bits2Shifts)res = ((res & bits2HMask00)>>1u) | (res & bits2LMask00);
  if(1  < bits2Shifts)res = ((res & bits2HMask01)>>1u) | (res & bits2LMask01);
  if(2  < bits2Shifts)res = ((res & bits2HMask02)>>1u) | (res & bits2LMask02);
  if(3  < bits2Shifts)res = ((res & bits2HMask03)>>1u) | (res & bits2LMask03);
  if(4  < bits2Shifts)res = ((res & bits2HMask04)>>1u) | (res & bits2LMask04);
  if(5  < bits2Shifts)res = ((res & bits2HMask05)>>1u) | (res & bits2LMask05);
  if(6  < bits2Shifts)res = ((res & bits2HMask06)>>1u) | (res & bits2LMask06);
  if(7  < bits2Shifts)res = ((res & bits2HMask07)>>1u) | (res & bits2LMask07);
  if(8  < bits2Shifts)res = ((res & bits2HMask08)>>1u) | (res & bits2LMask08);
  if(9  < bits2Shifts)res = ((res & bits2HMask09)>>1u) | (res & bits2LMask09);
  if(10 < bits2Shifts)res = ((res & bits2HMask10)>>1u) | (res & bits2LMask10);

  if(bits1Count != 0)
    res = uint(res & bits1DstMask) | uint((v[longestAxis]&bits1SrcMask)<<bits1SrcShift);

  return res;
}

).";

const std::string rssv::demortonShader = R".(
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

// m - length of 3 bits together
// n - length of 2 bits together
// o - length of 1 bit  alone
//
//                      bits1Count|bits2shifts|1bit_offset|2bit_offset
// ..*|   .**|.**|.**|       o    |     n-1   |   2n+3m   |  2i+2+3m
// .*.|   .**|.**|.**|       o    |     n-1   |   2n+3m   |  2i+2+3m
// ...|   .**|.**|.**|       o    |     n-1   |   2n+3m   |  2i+2+3m
// ###################################################################
// ..*|   *.*|*.*|*.*|       o-1  |     n     |   2n+3m+1 |  2i+1+3m
// *..|   *.*|*.*|*.*|       o+1  |     n-1   |   2n+3m-1 |  2i+1+3m
// ...|   *.*|*.*|*.*|       o    |     n     |   2n+3m   |  2i+1+3m
// ###################################################################
// .*.|   **.|**.|**.|       o    |     n     |   2n+3m   |  2i+0+3m
// *..|   **.|**.|**.|       o    |     n     |   2n+3m   |  2i+0+3m
// ...|   **.|**.|**.|       o    |     n     |   2n+3m   |  2i+0+3m
// ###################################################################
// ..*|   ...|...|...|       o    |     n     |   2n+3m   |  xxxxxxx
// .*.|   ...|...|...|       o    |     n     |   2n+3m   |  xxxxxxx
// ...|   ...|...|...|       o    |     n     |   2n+3m   |  xxxxxxx

uvec3 demorton2(uint v){
  uvec3 res = uvec3(0);
  uint counters[3] = {0,0,0};
  const uint limits[3] = {xBits,yBits,zBits};
  const uint allBits = xBits + yBits + zBits;
  uint a = 0;
  for(uint b=0;b<allBits;++b){
    res[a] |= ((v>>b)&1u) << counters[a];
    counters[a]++;
    a = (a+1u)%3u;
    if(counters[a] >= limits[a])a = (a+1u)%3u;
    if(counters[a] >= limits[a])a = (a+1u)%3u;
  }
  return res;
}

uvec3 demorton(uint res){
  const uint shortestZ     = uint(shortestAxis == 2u);
  const uint shortestY     = uint(shortestAxis == 1u);
  const uint isMiddle      = uint(bitTogether[1] > 0u);
  const uint isLongest     = uint(bitTogether[2] > 0u);
  const uint longestZ      = uint(longestAxis == 2u) * isLongest;
  const uint longestX      = uint(longestAxis == 0u) * isLongest;

  const uint bits2Shifts   = uint(uint(bitTogether[1] - uint(shortestZ | (shortestY & longestZ))) * isMiddle);
  
  const uint bits2OffsetB   = bitTogether[0]*3u + shortestAxis;
  const uint bits2Offset00  = bits2OffsetB + 2* 0;
  const uint bits2Offset01  = bits2OffsetB + 2* 1;
  const uint bits2Offset02  = bits2OffsetB + 2* 2;
  const uint bits2Offset03  = bits2OffsetB + 2* 3;
  const uint bits2Offset04  = bits2OffsetB + 2* 4;
  const uint bits2Offset05  = bits2OffsetB + 2* 5;
  const uint bits2Offset06  = bits2OffsetB + 2* 6;
  const uint bits2Offset07  = bits2OffsetB + 2* 7;
  const uint bits2Offset08  = bits2OffsetB + 2* 8;
  const uint bits2Offset09  = bits2OffsetB + 2* 9;
  const uint bits2Offset10  = bits2OffsetB + 2*10;

  const uint bits2LMask00 = uint((1u << bits2Offset00)-1u);
  const uint bits2LMask01 = uint((1u << bits2Offset01)-1u);
  const uint bits2LMask02 = uint((1u << bits2Offset02)-1u);
  const uint bits2LMask03 = uint((1u << bits2Offset03)-1u);
  const uint bits2LMask04 = uint((1u << bits2Offset04)-1u);
  const uint bits2LMask05 = uint((1u << bits2Offset05)-1u);
  const uint bits2LMask06 = uint((1u << bits2Offset06)-1u);
  const uint bits2LMask07 = uint((1u << bits2Offset07)-1u);
  const uint bits2LMask08 = uint((1u << bits2Offset08)-1u);
  const uint bits2LMask09 = uint((1u << bits2Offset09)-1u);
  const uint bits2LMask10 = uint((1u << bits2Offset10)-1u);

  const uint bits2HMask00 = (~bits2LMask00)<<1u;
  const uint bits2HMask01 = (~bits2LMask01)<<1u;
  const uint bits2HMask02 = (~bits2LMask02)<<1u;
  const uint bits2HMask03 = (~bits2LMask03)<<1u;
  const uint bits2HMask04 = (~bits2LMask04)<<1u;
  const uint bits2HMask05 = (~bits2LMask05)<<1u;
  const uint bits2HMask06 = (~bits2LMask06)<<1u;
  const uint bits2HMask07 = (~bits2LMask07)<<1u;
  const uint bits2HMask08 = (~bits2LMask08)<<1u;
  const uint bits2HMask09 = (~bits2LMask09)<<1u;
  const uint bits2HMask10 = (~bits2LMask10)<<1u;

  const uint bits1Count    = uint(bitTogether[2] - uint(shortestY & longestX) + uint(shortestY & longestZ)) * isLongest;
  const uint bits1used     = bitLength[2] - bits1Count;
  const uint bits1DstMask  = uint((1u<<(bitTogether[0]*3u + bitTogether[1]*2u + uint(shortestY & longestX) - uint(longestZ & shortestY))) -1u);
  const uint bits1SrcShift = bitTogether[0]*3u + bitTogether[1]*2u - uint(shortestY & longestZ) + uint(shortestY & longestX)  - bits1used;
  const uint bits1SrcMask  = ~((1u<<bits1used)-1u);

  uvec3 v = uvec3(0);

  uint last = 0;
  if(bits1Count != 0){
    last |= (res >> bits1SrcShift) & bits1SrcMask;
    res &= bits1DstMask;
  }

  if(10 < bits2Shifts)res = ((res<<1u)&bits2HMask10) | (res & bits2LMask10);
  if(9  < bits2Shifts)res = ((res<<1u)&bits2HMask09) | (res & bits2LMask09);
  if(8  < bits2Shifts)res = ((res<<1u)&bits2HMask08) | (res & bits2LMask08);
  if(7  < bits2Shifts)res = ((res<<1u)&bits2HMask07) | (res & bits2LMask07);
  if(6  < bits2Shifts)res = ((res<<1u)&bits2HMask06) | (res & bits2LMask06);
  if(5  < bits2Shifts)res = ((res<<1u)&bits2HMask05) | (res & bits2LMask05);
  if(4  < bits2Shifts)res = ((res<<1u)&bits2HMask04) | (res & bits2LMask04);
  if(3  < bits2Shifts)res = ((res<<1u)&bits2HMask03) | (res & bits2LMask03);
  if(2  < bits2Shifts)res = ((res<<1u)&bits2HMask02) | (res & bits2LMask02);
  if(1  < bits2Shifts)res = ((res<<1u)&bits2HMask01) | (res & bits2LMask01);
  if(0  < bits2Shifts)res = ((res<<1u)&bits2HMask00) | (res & bits2LMask00);

  v[2] = (res & 0x24924924u)>>2u;
  v[1] = (res & 0x92492492u)>>1u;
  v[0] = (res & 0x49249249u)>>0u;

  v[2] = (v[2] | (v[2]>> 2u)) & 0xc30c30c3u;
  v[2] = (v[2] | (v[2]>> 4u)) & 0x0f00f00fu;
  v[2] = (v[2] | (v[2]>> 8u)) & 0xff0000ffu;
  v[2] = (v[2] | (v[2]>>16u)) & 0x0000ffffu;

  v[1] = (v[1] | (v[1]>> 2u)) & 0xc30c30c3u;
  v[1] = (v[1] | (v[1]>> 4u)) & 0x0f00f00fu;
  v[1] = (v[1] | (v[1]>> 8u)) & 0xff0000ffu;
  v[1] = (v[1] | (v[1]>>16u)) & 0x0000ffffu;

  v[0] = (v[0] | (v[0]>> 2u)) & 0xc30c30c3u;
  v[0] = (v[0] | (v[0]>> 4u)) & 0x0f00f00fu;
  v[0] = (v[0] | (v[0]>> 8u)) & 0xff0000ffu;
  v[0] = (v[0] | (v[0]>>16u)) & 0x0000ffffu;
 
  if(bits1Count != 0){
  v[longestAxis] |= last;
  }

  return v;
}

).";
