#include <BitSet.h>

#include <cassert>
#include <algorithm>
#include <functional>

constexpr uint32_t BYTE_BITS = 8u;

BitSet::BitSet(unsigned int nofBits)
{
	assert(nofBits > 0);

	NofBits = nofBits;
	NofBytes = NofBits / BYTE_BITS;
	NofBytes += NofBits % BYTE_BITS == 0 ? 0 : 1;

	BitArray = std::make_unique<uint8_t[]>(NofBytes);

	clear();
}

BitSet::BitSet(BitSet const& other)
{
	copyOther(other);
}

BitSet::BitSet(BitSet&& other)
{
	NofBits = other.NofBits;
	NofBytes = other.NofBytes;

	BitArray = std::move(other.BitArray);
}

BitSet::~BitSet()
{
}

void BitSet::clear()
{
	for (unsigned int i = 0; i < NofBytes; ++i)
	{
		BitArray[i] = 0;
	}
}

void BitSet::setBit(unsigned int pos)
{
	if(pos>=NofBits)
	{
		return;
	}

	uint32_t const bytePos = pos / BYTE_BITS;
	uint32_t const bitPos = pos % BYTE_BITS;

	BitArray[bytePos] |= 1 << bitPos;
}

void BitSet::clearBit(unsigned int pos)
{
	if (pos >= NofBits)
	{
		return;
	}

	uint32_t const bytePos = pos / BYTE_BITS;
	uint32_t const bitPos = pos % BYTE_BITS;

	BitArray[bytePos] &= ~uint8_t(1 << bitPos);
}

bool BitSet::getBit(unsigned int pos) const
{
	if (pos >= NofBits)
	{
		return false;
	}

	uint32_t const bytePos = pos / BYTE_BITS;
	uint32_t const bitPos = pos % BYTE_BITS;

	return BitArray[bytePos] >> bitPos & 1;
}

void BitSet::operator=(const BitSet& other)
{
	copyOther(other);
}

size_t BitSet::hash() const
{
	if (NofBits == BYTE_BITS)
	{
		return hash8();
	}
	else if (NofBits == BYTE_BITS * BYTE_BITS)
	{
		return hash64();
	}

	return hashGeneric();
}

void BitSet::copyOther(BitSet const& other)
{
	NofBits = other.NofBits;
	NofBytes = other.NofBytes;

	BitArray = std::make_unique<uint8_t[]>(NofBytes);

	memcpy(&BitArray[0], &other.BitArray[0], NofBytes);
}

bool BitSet::compare8(uint8_t other) const
{
	return BitArray[0] == other;
}

bool BitSet::compare64(uint64_t other) const
{
	return getAs<uint64_t>() == other;
}

bool BitSet::compareGeneric(BitSet const& other) const
{
	for (uint32_t i = 0; i < NofBytes; ++i)
	{
		if(BitArray[i]!=other.BitArray[i])
		{
			return false;
		}
	}

	return true;
}

size_t BitSet::hash8() const
{
	return BitArray[0];
}

size_t BitSet::hash64() const
{
	return getAs<size_t>();
}

size_t BitSet::hashGeneric() const
{
	size_t seed = 0;
	auto combineHash = [&seed](uint8_t val)
	{
		seed ^= std::hash<uint8_t>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	};
	
	for(unsigned int i = 0; i< NofBytes; ++i)
	{
		combineHash(BitArray[i]);
	}

	return seed;
}

bool BitSet::operator==(const BitSet& other) const
{
	if(NofBits != other.NofBits)
	{
		return false;
	}

	if(NofBits==BYTE_BITS)
	{
		return compare8(other.BitArray[0]);
	}
	else if (NofBits == BYTE_BITS * BYTE_BITS)
	{
		return compare64(other.getAs<uint64_t>());
	}
	
	return compareGeneric(other);
}

