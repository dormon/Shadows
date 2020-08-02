#include <TSSV.hpp>
#include <SidesShaderGenerator.hpp>
#include <GSCaps.h>

#include <geGL/Buffer.h>

#include <Vars/Resource.h>
#include <FunctionPrologue.h>
#include <createAdjacency.h>

using namespace ge::gl;

TSSV::TSSV(vars::Vars& vars) : ShadowVolumes(vars)
{ 
}

TSSV::~TSSV()
{
	vars.erase("tssv.objects");
}

void TSSV::drawSides(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	createAdjacency(vars);
	createProgram();
	createVertexBuffer();
	createElementBuffer();
	createVertexArray();
	setCounts();

	const glm::mat4 mvp = projectionMatrix * viewMatrix;

	vars.get<VertexArray>("tssv.objects.VAO")->bind();

	Program* program = vars.get<Program>("tssv.objects.program");
	program->use();
	program->setMatrix4fv("mvp", glm::value_ptr(mvp), 1,GL_FALSE);
	program->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);

	glPatchParameteri(GL_PATCH_VERTICES, GLint(_patchVertices));
	glDrawElements(GL_PATCHES, GLsizei(_patchVertices*_nofEdges), GL_UNSIGNED_INT, nullptr);

	vars.get<VertexArray>("tssv.objects.VAO")->unbind();
}

void TSSV::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	createCapsDrawer();
	 
	vars.get<GSCaps>("tssv.objects.capsDrawer")->drawCaps(lightPosition, viewMatrix, projectionMatrix);
}

void TSSV::createVertexBuffer()
{
	FUNCTION_PROLOGUE("tssv.objects", "model", "adjacency");

	Adjacency const* ad = vars.get<Adjacency>("adjacency");

	Buffer* vbo = vars.reCreate<Buffer>("tssv.objects.VBO", sizeof(float) * 4 * (ad->getNofTriangles() * 3 + ad->getNofEdges()));
	
	float*ptr = (float*)vbo->map();

	for (unsigned p = 0; p<ad->getNofTriangles() * 3; ++p) 
	{
		//loop over points
		for (unsigned e = 0; e<3; ++e)
			ptr[p * 4 + e] = ad->getVertices()[p * 3 + e];
		ptr[p * 4 + 3] = 1;
	}

	for (unsigned e = 0; e<ad->getNofEdges(); ++e)
		ptr[(ad->getNofTriangles() * 3 + e) * 4 + 0] = float(ad->getNofOpposite(e));
	
	vbo->unmap();
}

void TSSV::createElementBuffer()
{
	FUNCTION_PROLOGUE("tssv.objects", "model", "adjacency");

	Adjacency const* ad = vars.get<Adjacency>("adjacency");

	size_t patchVertices = getNofPatchVertices();

	Buffer* ebo = vars.reCreate<Buffer>("tssv.objects.EBO", sizeof(unsigned) * patchVertices * ad->getNofEdges());

	unsigned* eptr = (unsigned*)ebo->map();
	for (unsigned e = 0; e < ad->getNofEdges(); ++e)
	{
		//loop over edges
		unsigned base = e * unsigned(patchVertices);
		eptr[base + 0] = unsigned(ad->getEdge(e, 0)) / 3;
		eptr[base + 1] = unsigned(ad->getEdge(e, 1)) / 3;
		eptr[base + 2] = unsigned(ad->getNofTriangles()) * 3 + e;
		for (unsigned o = 0; o < ad->getMaxMultiplicity(); ++o)
			if (o < ad->getNofOpposite(e))
				eptr[base + 3 + o] = unsigned(ad->getOpposite(e, o)) / 3;
			else eptr[base + 3 + o] = 0;
	}

	ebo->unmap();
}

void TSSV::createVertexArray()
{
	FUNCTION_PROLOGUE("tssv.objects", "model", "adjacency");
	
	VertexArray* vao = vars.reCreate<VertexArray>("tssv.objects.VAO");

	vao->addAttrib(vars.get<Buffer>("tssv.objects.VBO"), 0, 4, GL_FLOAT);
	vao->addElementBuffer(vars.get<Buffer>("tssv.objects.EBO"));
}

void TSSV::createProgram()
{
	FUNCTION_PROLOGUE("tssv.objects", "maxMultiplicity", "tssv.args.useRefEdge", "tssv.args.cullSides", "tssv.args.useStencilExport");

	STSSilTemplate TTS;
	TTS.Version = 430;
	TTS.UseLayouts = true;
	TTS.Universal = true;
	TTS.UseSillyPerPatchLevel = true;
	TTS.UseOptimizedDegeneration = true;
	TTS.UseCompatibility = false;
	TTS.LightPositionUniformName = "LightPosition";
	TTS.MatrixUniformName = "mvp";
	TTS.VertexAttribName = "Position";
	TTS.UseReferenceEdge = vars.getBool("tssv.args.useRefEdge");;
	TTS.CullSides = vars.getBool("tssv.args.cullSides");;
	TTS.UseStencilValueExport = vars.getBool("tssv.args.useStencilExport");;

	unsigned int const maxMult = vars.getUint32("maxMultiplicity");

	std::shared_ptr<ge::gl::Shader>TSSilVerHull2 = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, GenTSSilVertexHull(TTS));
	std::shared_ptr<ge::gl::Shader>TSSilConHull2 = std::make_shared<ge::gl::Shader>(GL_TESS_CONTROL_SHADER, GenTSSilControlHull(maxMult, TTS));
	std::shared_ptr<ge::gl::Shader>TSSilEvaHull2 = std::make_shared<ge::gl::Shader>(GL_TESS_EVALUATION_SHADER, GenTSSilEvaluationHull(TTS));
	std::shared_ptr<ge::gl::Shader>TSSilFragHull2 = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, GenTSSilFragmentHull(TTS));
	
	vars.reCreate<Program>("tssv.objects.program", TSSilVerHull2, TSSilConHull2, TSSilEvaHull2, TSSilFragHull2);
}

void TSSV::createCapsDrawer()
{
	FUNCTION_PROLOGUE("tssv.objects", "model", "maxMultiplicity");

	vars.reCreate<GSCaps>("tssv.objects.capsDrawer", vars);
}

void TSSV::setCounts()
{
	FUNCTION_PROLOGUE("tssv.objects", "model", "adjacency");
	
	Adjacency const* ad = vars.get<Adjacency>("adjacency");

	_patchVertices = getNofPatchVertices();
	_nofEdges = ad->getNofEdges();
}

size_t TSSV::getNofPatchVertices() const
{
	Adjacency const* ad = vars.get<Adjacency>("adjacency");
	return 2 + 1 + ad->getMaxMultiplicity();
}

