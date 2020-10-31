#include <OctreeSerializer.h>
#include <Octree.h>

#include <glm/gtc/type_ptr.inl>

#include <iostream>
#include <sstream>
#include <functional>

//"fopen might be unsafe" warning
#pragma warning(disable : 4996)

std::string OctreeSerializer::GenerateFileName(SerializerData const& data) const
{
	size_t const hashedParams = HashParams(data);

	return data.modelName + "_" + std::to_string(hashedParams) + ".hssv";
}

bool OctreeSerializer::loadFromFile(Octree* octree, SerializerData const& data)
{
	std::string const octreeFile = GenerateFileName(data);
	std::cout << "Opening stored octree " << octreeFile << std::endl;

	FILE* input = fopen(octreeFile.c_str(), "rb");
	if (!input)
	{
		return false;
	}

	//Read num levels
	u32 const numLevels = ReadUint32(input);
  (void)numLevels;

	//Read root AABB
	AABB const rootVolume = ReadAabb(input);
  (void)rootVolume;

	//Read node data
	u32 const numNodes = octree->getTotalNumNodes();

	for (u32 i = 0; i < numNodes; ++i)
	{
		{
			//Read number of potential subbuffers
			u32 const numSubbuffersPotential = ReadUint32(input);

			Node* node = octree->getNode(i);

			for (u32 bufferId = 0; bufferId < numSubbuffersPotential; ++bufferId)
			{
				//Read subbuffer id
				u8 const subbufferId = ReadUint8(input);

				//Read subbuffer size
				u32 const subbufferSize = ReadUint32(input);

				//Read subbuffer data
				ReadUint32Buffer(input, subbufferSize, node->edgesMayCastMap[subbufferId]);
			}
		}

		{
			//Read number of silhouette subbuffers
			u32 const numSubbuffersSilhouette = ReadUint32(input);

			Node* node = octree->getNode(i);

			for (u32 bufferId = 0; bufferId < numSubbuffersSilhouette; ++bufferId)
			{
				//Read subbuffer id
				u8 const subbufferId = ReadUint8(input);

				//Read subbuffer size
				u32 const subbufferSize = ReadUint32(input);

				//Read subbuffer data
				ReadUint32Buffer(input, subbufferSize, node->edgesAlwaysCastMap[subbufferId]);
			}
		}
	}

	fclose(input);

	return octree;
}

void OctreeSerializer::storeToFile(Octree* octree, SerializerData const& data)
{
	FILE* output = fopen(GenerateFileName(data).c_str(), "wb");
	if (!output)
	{
		return;
	}

	//Write octree level
	WriteUint32(output, octree->getDeepestLevel());

	//Write top level AABB
	WriteAabb(output, octree->getNode(0)->volume);

	//Write nodes
	u32 const nofNodes = octree->getTotalNumNodes();
	for (u32 i = 0; i<nofNodes; ++i)
	{
		auto node = octree->getNode(i);

		{
			//Write nof potential subbuffers
			WriteUint32(output, u32(node->edgesMayCastMap.size()));

			for(auto const& subBuffer : node->edgesMayCastMap)
			{
				//Write subbuffer Id (bitmask)
				WriteUint8(output, subBuffer.first);

				//write subbuffer size
				WriteUint32(output, u32(subBuffer.second.size()));

				//write data
				WriteUint32Buffer(output, subBuffer.second);
			}
		}

		{
			//Write nof silhouette subbuffers
			WriteUint32(output, u32(node->edgesAlwaysCastMap.size()));

			for (auto const& subBuffer : node->edgesAlwaysCastMap)
			{
				//Write subbuffer Id (bitmask)
				WriteUint8(output, subBuffer.first);

				//write subbuffer size
				WriteUint32(output, u32(subBuffer.second.size()));

				//write data
				WriteUint32Buffer(output, subBuffer.second);
			}
		}

	}

	//Finish
	fclose(output);
}

u32 OctreeSerializer::ReadUint32(FILE* f)
{
	u32 n = 0;
	fread(&n, sizeof(u32), 1, f);
	return n;
}

u8 OctreeSerializer::ReadUint8(FILE* f)
{
	u8 n = 0;
	fread(&n, sizeof(u8), 1, f);
	return n;
}

AABB OctreeSerializer::ReadAabb(FILE* f)
{
	glm::vec3 minP, maxP;
	AABB bbox;

	fread(glm::value_ptr(minP), sizeof(float), 3, f);
	fread(glm::value_ptr(maxP), sizeof(float), 3, f);

	bbox.setMin(minP);
	bbox.setMax(maxP);

	return bbox;
}

void OctreeSerializer::ReadUint32Buffer(FILE* f, u32 nofUints, std::vector<u32>& buffer)
{
	if (!nofUints)
	{
		return;
	}
	
	buffer.resize(nofUints);

	fread(buffer.data(), sizeof(u32), nofUints, f);
}

void OctreeSerializer::WriteUint32(FILE* output, u32 value)
{
	fwrite(&value, sizeof(u32), 1, output);
}

void OctreeSerializer::WriteUint8(FILE* output, u8 value)
{
	fwrite(&value, sizeof(u8), 1, output);
}

void OctreeSerializer::WriteAabb(FILE* output, const AABB& bbox)
{
	fwrite(glm::value_ptr(bbox.getMin()), sizeof(float), 3, output);
	fwrite(glm::value_ptr(bbox.getMax()), sizeof(float), 3, output);
}

void OctreeSerializer::WriteUint32Buffer(FILE* output, const std::vector<u32>& buffer)
{
	if (buffer.empty())
	{
		return;
	}

	fwrite(buffer.data(), sizeof(u32), buffer.size(), output);
}

size_t OctreeSerializer::HashParams(SerializerData const& data) const
{
	size_t seed = 0;
	
	combineHash(seed, data.modelName);
	combineHash(seed, data.sceneScale);
	combineHash(seed, data.deepestLevel);
	combineHash(seed, data.isCompressed);
	combineHash(seed, data.minNofEdgesInSubnodes);

	return seed;
}
