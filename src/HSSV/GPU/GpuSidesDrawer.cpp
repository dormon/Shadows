#include <GPU/GpuSidesDrawer.h>
#include <Octree.h>
#include <AdjacencyWrapper.h>

#include <FastAdjacency.h>


using namespace ge::gl;

GpuSidesDrawer::GpuSidesDrawer(Octree* o, Adjacency* ad, u32 maxMultiplicity) : SidesDrawerBase(o)
{
	MaxMultiplicity = maxMultiplicity;
	Ad = ad;
}

void GpuSidesDrawer::CreateBuffers()
{
	CreateEdgeBuffer();
	CreateOctreeBuffer();
}

void GpuSidesDrawer::CreateEdgeBuffer()
{
	size_t const nofEdges = Ad->getNofEdges();
	edgesBuffer = std::make_unique<Buffer>(nofEdges * (3 + 3 + 1 + MaxMultiplicity * 3) * sizeof(float));

	float * mappedBuf = reinterpret_cast<float*>(edgesBuffer->map(GL_WRITE_ONLY));
	size_t const edgeSize = 7 + 3 * MaxMultiplicity;

	for(size_t edge = 0; edge < nofEdges; ++edge)
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
