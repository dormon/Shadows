#pragma once

#include <Defines.h>

class MultiplicityCoder
{
public:
	MultiplicityCoder(u32 bitsPerMultiplicity);

	u32 encodeEdgeMultiplicityToId(u32 edgeID, s32 multiplicity);
	s32 decodeEdgeMultiplicityFromId(u32 edgeWithEncodedMultiplicity);
	u32 decodeEdgeFromEncoded(u32 edgeWithEncodedMultiplicity);

private:
	u32 BitsPerMultiplicity;
	u32 BitMask;
	u32 MaxAbsMultiplicity;
};
