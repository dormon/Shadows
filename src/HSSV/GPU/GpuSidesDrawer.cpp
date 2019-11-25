#include <GPU/GpuSidesDrawer.h>
#include <Octree.h>
#include <AdjacencyWrapper.h>

#include <FastAdjacency.h>


using namespace ge::gl;

GpuSidesDrawer::GpuSidesDrawer(Octree* o, Adjacency* ad, u32 maxMultiplicity) : SidesDrawerBase(o)
{
	MaxMultiplicity = maxMultiplicity;
	Ad = ad;

	CalcBitMasks8(2);
}

GpuSidesDrawer::~GpuSidesDrawer()
{

}

void GpuSidesDrawer::drawSides(const glm::mat4& mvp, const glm::vec4& light)
{

}

void GpuSidesDrawer::CreateBuffers()
{
	CreateEdgeBuffer();
	CreateOctreeBuffer();
}

void GpuSidesDrawer::CreateEdgeBuffer()
{
	u32 const nofEdges = u32(Ad->getNofEdges());
	edgesBuffer = std::make_unique<Buffer>(nofEdges * (3 + 3 + 1 + MaxMultiplicity * 3) * sizeof(float));

	float * mappedBuf = reinterpret_cast<float*>(edgesBuffer->map(GL_WRITE_ONLY));
	u32 const edgeSize = 7 + 3 * MaxMultiplicity;

	for(u32 edge = 0; edge < nofEdges; ++edge)
	{
		glm::vec3 const& v1 = getEdgeVertexLow(Ad, edge);
		glm::vec3 const& v2 = getEdgeVertexHigh(Ad, edge);
		u32 const nofOpposite = getNofOppositeVertices(Ad, edge);

		mappedBuf[edge * edgeSize + 0] = v1.x;
		mappedBuf[edge * edgeSize + 1] = v1.x;
		mappedBuf[edge * edgeSize + 2] = v1.x;

		mappedBuf[edge * edgeSize + 3] = v1.x;
		mappedBuf[edge * edgeSize + 4] = v1.x;
		mappedBuf[edge * edgeSize + 5] = v1.x;

		reinterpret_cast<u32*>(mappedBuf)[edge * edgeSize + 6] = nofOpposite;

		for(u32 opposite = 0; opposite < nofOpposite; ++opposite)
		{
			glm::vec3 const& o = getOppositeVertex(Ad, edge, opposite);

			mappedBuf[edge * edgeSize + 7 + 3 * opposite + 0] = o.x;
			mappedBuf[edge * edgeSize + 7 + 3 * opposite + 1] = o.y;
			mappedBuf[edge * edgeSize + 7 + 3 * opposite + 2] = o.z;
		}
	}

	edgesBuffer->unmap();
}

void GpuSidesDrawer::CreateOctreeBuffer()
{
	u32 const nofNodes = octree->getTotalNumNodes();
	octreeBuffer = std::make_unique<Buffer>();
}

u32 GpuSidesDrawer::GetNofIndicesPerBitmask() const
{
	//m_bitMasks[0].size() - because all bitmask arrays are of same size
	return u32(BitmasksWithIBitSet[0].size());
}

void GpuSidesDrawer::CalcBitMasks8(unsigned int minBits)
{
	BitmasksWithIBitSet.resize(8);

	for (uint32_t i = 1; i < 256; ++i)
	{
		BitSet8 num = i;
		if (num.count() < minBits)
		{
			continue;
		}

		for (uint32_t b = 0; b < 8; ++b)
		{
			if (num[b])
			{
				BitmasksWithIBitSet[b].push_back(u8(num.to_ulong()));
			}
		}
	}
}
