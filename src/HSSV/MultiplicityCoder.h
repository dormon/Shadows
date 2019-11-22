#pragma once

#include <stdint.h>

class MultiplicityCoder
{
public:
	MultiplicityCoder(uint32_t bitsPerMultiplicity);

	int32_t encodeEdgeMultiplicityToId(uint32_t edgeID, int multiplicity);
	int32_t decodeEdgeMultiplicityFromId(int32_t edgeWithEncodedMultiplicity);
	int32_t decodeEdgeFromEncoded(int32_t edgeWithEncodedMultiplicity);

private:
	uint32_t BitsPerMultiplicity;
	int32_t BitMask;
	int32_t MaxAbsMultiplicity;
};
