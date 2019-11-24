#include <CapsDrawer/HssvCapsDrawer.h>
#include <Defines.h>
#include <AdjacencyWrapper.h>
#include <CapsDrawer/HssvCapsDrawerShaders.h>

#include <FastAdjacency.h>

#include <geGL/StaticCalls.h>

using namespace ge::gl;

HssvCapsDrawer::HssvCapsDrawer(Adjacency* ad)
{
	Ad = ad;
	InitCapsPrograms();
	InitCapsBuffers();
}

HssvCapsDrawer::~HssvCapsDrawer()
{
}

void HssvCapsDrawer::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	const glm::mat4 mvp = projectionMatrix * viewMatrix;

	VAO->bind();

	SidesProgram->use();
	SidesProgram->setMatrix4fv("mvp", glm::value_ptr(mvp), 1, GL_FALSE);
	SidesProgram->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);

	glDrawArrays(GL_TRIANGLES, 0, GLsizei(NofCapsTriangles * 3));

	VAO->unbind();
}

void HssvCapsDrawer::InitCapsBuffers()
{
	NofCapsTriangles = 0;

	VAO = std::make_unique<VertexArray>();
	/*
	size_t maxSizeVbo = Ad->getNofEdges() * Ad->getMaxMultiplicity() * 3 * 4;

	std::vector<float> data;
	data.reserve(maxSizeVbo);

	u32 const nofEdges = u32(Ad->getNofEdges());
	
	for (u32 edge = 0; edge < nofEdges; ++edge)
	{
		glm::vec3 const& A = getEdgeVertexLow(Ad, edge);
		glm::vec3 const& B = getEdgeVertexHigh(Ad, edge);

		u32 const nofOpposite = getNofOppositeVertices(Ad, edge);

		NofCapsTriangles += nofOpposite;

		for(u32 i = 0; i<nofOpposite; ++i)
		{
			glm::vec3 const& C = getOppositeVertex(Ad, edge, i);

			data.push_back(A.x); data.push_back(A.y); data.push_back(A.z); data.push_back(1.f);
			data.push_back(B.x); data.push_back(B.y); data.push_back(B.z); data.push_back(1.f);
			data.push_back(C.x); data.push_back(C.y); data.push_back(C.z); data.push_back(1.f);
		}
	}

	VBO = std::make_unique<Buffer>(data.size() * sizeof(float), nullptr, GL_STATIC_DRAW);
	VAO->addAttrib(VBO, 0, 4, GL_FLOAT);
	*/

	//--
	NofCapsTriangles = Ad->getNofTriangles();
	VBO = std::make_unique<Buffer>(Ad->getVertices().size() * sizeof(float), Ad->getVertices().data(), GL_STATIC_DRAW);
	VAO->addAttrib(VBO, 0, 3, GL_FLOAT);
	//--
}

void HssvCapsDrawer::InitCapsPrograms()
{
	std::shared_ptr<ge::gl::Shader> vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, vsCapsHssv);
	std::shared_ptr<ge::gl::Shader> gs = std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER, gsCapsHssv);

	SidesProgram = std::make_unique<Program>(vs, gs);
}
