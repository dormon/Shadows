#pragma once

#include <memory>

class BitSet
{
public:

	BitSet(unsigned int nofBits);
	BitSet(BitSet const& other);
	BitSet(BitSet&& other);
	
	~BitSet();

	void clear();

	void setBit(unsigned int pos);
	void clearBit(unsigned int pos);
	bool getBit(unsigned int pos) const;

	bool operator==(BitSet const& other) const;

	void operator=(BitSet const& other);

	template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
	T getAs() const
	{
		return *reinterpret_cast<T*>(BitArray.get());
	} 

	size_t hash() const;

private:
	void copyOther(BitSet const& other);

	bool compare8(uint8_t other) const;
	bool compare64(uint64_t other) const;
	bool compareGeneric(BitSet const& other) const;

	size_t hash8() const;
	size_t hash64() const;
	size_t hashGeneric() const;

	unsigned int NofBits;
	unsigned int NofBytes;
	std::unique_ptr<uint8_t[]> BitArray;
};

class BitSetHashFunction 
{
public:

	size_t operator()(const BitSet& b) const
	{
		return b.hash();
	}
};