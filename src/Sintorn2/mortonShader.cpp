#include <Sintorn2/mortonShader.h>

const std::string sintorn2::mortonShader = R".(
#ifndef WARP
#define WARP 64
#endif//WARP

#ifndef WINDOW_X
#define WINDOW_X 512
#endif//WINDOW_X

#ifndef WINDOW_Y
#define WINDOW_Y 512
#endif//WINDOW_Y

#ifndef MIN_Z
#define MIN_Z 0
#endif//MIN_Z

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

uint morton(uvec3 v){
  const uint warpBits      = uint(ceil(log2(float(WARP))));
  const uint warpBitsX     = uint(warpBits/2u) + uint(warpBits%2 != 0u);
  const uint warpBitsY     = uint(warpBits-warpBitsX);
  const uint warpX         = uint(1u<<warpBitsX);
  const uint warpY         = uint(1u<<warpBitsY);
  const uint clustersX     = uint(WINDOW_X/warpX) + uint(WINDOW_X%warpX != 0u);
  const uint clustersY     = uint(WINDOW_Y/warpY) + uint(WINDOW_Y%warpY != 0u);
  const uint xBits         = uint(ceil(log2(float(clustersX))));
  const uint yBits         = uint(ceil(log2(float(clustersY))));
  const uint zBits         = MIN_Z>0?MIN_Z:max(max(xBits,yBits),MIN_Z);
  const uint shortest      = min(min(xBits,yBits),zBits);
  const uint middle        = max(max(min(xBits,yBits),min(xBits,zBits)),min(yBits,zBits));
  const uint longest       = max(max(xBits,yBits),zBits);
  const uint bits3Length   = shortest;
  const uint bits2Length   = uint(middle-shortest);
  const uint bits1Length   = uint(longest-middle);
  const uint shortestAxis  = uint(uint(shortest == yBits) + uint(shortest == zBits)*2u);
  const uint longestAxis   = uint(uint(longest  == yBits) + uint(longest  == zBits)*2u);
  const uint shortestZ     = uint(shortestAxis == 2u);
  const uint shortestY     = uint(shortestAxis == 1u);
  const uint longestZ      = uint(longestAxis == 2u);
  const uint longestX      = uint(longestAxis == 0u);
  const uint isMiddle      = uint(middle > 0);
  const uint isLongest     = uint(longest > 0);
  const uint bits2Shifts   = uint(uint(bits2Length - uint(shortestZ | (shortestY & longestZ))) * isMiddle);

  const uint bits2OffsetB   = bits3Length*3u + shortestAxis;
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

  const uint bits2HMask00 = ~bits2LMask00;
  const uint bits2HMask01 = ~bits2LMask01;
  const uint bits2HMask02 = ~bits2LMask02;
  const uint bits2HMask03 = ~bits2LMask03;
  const uint bits2HMask04 = ~bits2LMask04;
  const uint bits2HMask05 = ~bits2LMask05;
  const uint bits2HMask06 = ~bits2LMask06;
  const uint bits2HMask07 = ~bits2LMask07;
  const uint bits2HMask08 = ~bits2LMask08;
  const uint bits2HMask09 = ~bits2LMask09;
  const uint bits2HMask10 = ~bits2LMask10;

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
  
  const uint bits1Count    = uint(bits1Length - uint(shortestY & longestX) + uint(shortestY & longestZ)) * isLongest;
  const uint bits1used     = longest - bits1Count;
  const uint bits1DstMask  = uint((1u<<(bits3Length*3u + bits2Length*2u + uint(shortestY & longestX)/*- longestZ*/)) -1u);
  const uint bits1SrcShift = bits3Length*3u + bits2Length*2u - uint(shortestY & longestZ) + uint(shortestY & longestX)  - bits1used;
  const uint bits1SrcMask  = ~((1u<<bits1used)-1u);


  if(bits1Count != 0)
    res = uint(res & bits1DstMask) | uint((v[longestAxis]&bits1SrcMask)<<bits1SrcShift);

  return res;
}

).";
