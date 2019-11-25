#pragma once

#include <Defines.h>
#include <AABB.h>

#include <string>
#include <vector>

#include <glm/glm.hpp>

class Octree;

class OctreeSerializer
{
public:
	bool loadFromFile(Octree* octree, std::string const& modelName, float sceneScale, bool isCompressed);
	void storeToFile( Octree* octree, std::string const& modelName, float sceneScale, bool isCompressed);

protected:
	std::string GenerateFileName(std::string const& modelFilename, float sceneScale, u32 deepestLevel, bool isCompressed) const;

	u32     ReadUint32(FILE* input);
	u8      ReadUint8(FILE* input);
	AABB	ReadAabb(FILE* input);
	void	ReadUint32Buffer(FILE* input, u32 nofUints, std::vector<u32>& buffer);

	void	WriteUint32(FILE* output, u32 value);
	void	WriteUint8(FILE* output, u8 value);
	void    WriteAabb(FILE* output, AABB const& bbox);
	void	WriteUint32Buffer(FILE* output, std::vector<u32> const& buffer);

	size_t HashParams(std::string const& modelFilename, float sceneScale, u32 deepestLevel, bool isCompressed) const;

	template<class T>
	void combineHash(size_t& seed, T const& val) const
	{
		seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

};
