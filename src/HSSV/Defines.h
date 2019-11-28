#pragma once

#include <stdint.h>
#include <bitset>

typedef uint64_t u64;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint8_t u8;
typedef int8_t s8;

constexpr u32 OCTREE_NUM_CHILDREN = 8u;
constexpr u8 BITMASK_ALL_SET = u8(-1);

typedef std::bitset<OCTREE_NUM_CHILDREN> BitSet8;