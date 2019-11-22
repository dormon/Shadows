#include <MultiplicityCoder.h>

#include <cassert>
#include <cstdlib>

inline int encodeEdgeMultiplicityToId(uint32_t edgeID, int multiplicity)
{
	assert(multiplicity != 0);
	assert(multiplicity <= 2 && multiplicity >= -2);

	int val = edgeID;

	val |= multiplicity < 0 ? (1 << 31) : 0;
	val |= abs(multiplicity) == 2 ? (1 << 30) : 0;

	return val;
}

inline int decodeEdgeMultiplicityFromId(int edgeWithEncodedMultiplicity)
{
	int val = 1;

	const int sign = (edgeWithEncodedMultiplicity & (1 << 31)) != 0 ? -1 : 1;
	const int isTwo = (edgeWithEncodedMultiplicity & (1 << 30)) != 0;

	return (val + isTwo) * sign;
}

inline int decodeEdgeFromEncoded(int edgeWithEncodedMultiplicity)
{
	return edgeWithEncodedMultiplicity & 0x3FFFFFFF;
}

MultiplicityCoder::MultiplicityCoder(uint32_t bitsPerMultiplicity)
{
	assert(BitsPerMultiplicity < 32);
	assert(BitsPerMultiplicity != 0);

	BitsPerMultiplicity = bitsPerMultiplicity;
	BitMask = ((int32_t)1 << BitsPerMultiplicity) - 1;
	MaxAbsMultiplicity = (1 << bitsPerMultiplicity-1)-1;
}

int32_t MultiplicityCoder::encodeEdgeMultiplicityToId(uint32_t edgeID, int32_t multiplicity)
{
	assert(abs(multiplicity) <= MaxAbsMultiplicity);
	assert(edgeID < (1 << (32- BitsPerMultiplicity)));
	assert(multiplicity != 0);

	return int32_t(edgeID << BitsPerMultiplicity) | (multiplicity & BitMask);
}

int32_t MultiplicityCoder::decodeEdgeMultiplicityFromId(int32_t edgeWithEncodedMultiplicity)
{
	return edgeWithEncodedMultiplicity & BitMask;
}

int32_t MultiplicityCoder::decodeEdgeFromEncoded(int32_t edgeWithEncodedMultiplicity)
{
	return edgeWithEncodedMultiplicity >> BitsPerMultiplicity;
}
