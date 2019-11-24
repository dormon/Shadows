#include <MultiplicityCoder.h>

#include <cassert>
#include <cstdlib>

MultiplicityCoder::MultiplicityCoder(u32 bitsPerMultiplicity)
{
	assert(bitsPerMultiplicity < 32);
	assert(bitsPerMultiplicity != 0);

	BitsPerMultiplicity = bitsPerMultiplicity;
	BitMask = ((u32)1 << BitsPerMultiplicity) - 1;
	MaxAbsMultiplicity = (1 << (bitsPerMultiplicity-1))-1;
}

u32 MultiplicityCoder::encodeEdgeMultiplicityToId(u32 edgeID, s32 multiplicity)
{
	assert(u32(abs(multiplicity)) <= MaxAbsMultiplicity);
	assert(edgeID < ((1u << (32u- BitsPerMultiplicity))-1) );
	assert(multiplicity != 0);

	return u32(edgeID << BitsPerMultiplicity) | (multiplicity & BitMask);
}

s32 MultiplicityCoder::decodeEdgeMultiplicityFromId(u32 edgeWithEncodedMultiplicity)
{
	s32 retval = s32(edgeWithEncodedMultiplicity & BitMask);
	//sign extension
	s32 m = 1u << (BitsPerMultiplicity - 1);
	return (retval ^ m) - m;
}

u32 MultiplicityCoder::decodeEdgeFromEncoded(u32 edgeWithEncodedMultiplicity)
{
	return edgeWithEncodedMultiplicity >> BitsPerMultiplicity;
}
