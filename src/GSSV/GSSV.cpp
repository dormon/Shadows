#include <GSSV.hpp>
#include <GenerateGeometryShaderSilhouette.hpp>
#include <GSCaps.h>
#include <Model.h>

#include <geGL/Buffer.h>

#include <Vars/Resource.h>
#include <FunctionPrologue.h>

using namespace ge::gl;

GSSV::GSSV( vars::Vars& vars) : ShadowVolumes(vars)
{
}

GSSV::~GSSV()
{
	vars.erase("gssv");
}

void GSSV::createCapsDrawer(vars::Vars& vars)
{
	FUNCTION_PROLOGUE("gssv", "model", "maxMultiplicity");

	vars.reCreate<GSCaps>("gssv.capsDrawer", _adjacency);
}

void GSSV::drawUser(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	auto t = glIsEnabled(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnable(GL_DEPTH_TEST);
    const glm::mat4 mvp = projectionMatrix * viewMatrix;

	vars.get<VertexArray>("gssv.sidesVAO")->bind();
	Program* program = vars.get<Program>("gssv.sidesVisualizationProgram");
    program->use();
    program->setMatrix4fv("mvp", glm::value_ptr(mvp), 1, GL_FALSE);
    program->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);

    glDrawArrays(GL_POINTS, 0, GLsizei(_nofEdges));

	vars.get<VertexArray>("gssv.sidesVAO")->unbind();

	GSCaps* capsDrawer = vars.get<GSCaps>("gssv.capsDrawer");

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
    capsDrawer->drawCapsVisualized(lightPosition, viewMatrix, projectionMatrix, true, false, true, false, glm::vec3(0, 1, 0));
	glDisable(GL_CULL_FACE);
	capsDrawer->drawCapsVisualized(lightPosition, viewMatrix, projectionMatrix, false, true, false, true, glm::vec3(0, 0, 1));

	glDisable(GL_DEPTH_CLAMP);
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void GSSV::createSidesVBO(vars::Vars& vars)
{
	FUNCTION_PROLOGUE("gssv", "model", "maxMultiplicity");

	Model* model = vars.get<Model>("model");
	std::vector<float> verts = model->getVertices();

	_nofEdges = _adjacency->getNofEdges();
	unsigned const NumV = getNofAttributes();

	Buffer* vbo = vars.reCreate<Buffer>("gssv.sidesVBO");
	vbo->alloc(sizeof(float) * 4 * NumV * _nofEdges);

	float* Ptr = (float*)vbo->map();
	std::vector<float> const vertices = _adjacency->getVertices();

	for (unsigned e = 0; e < _nofEdges; ++e)
	{
		//A
		for (int k = 0; k < 3; ++k)
			Ptr[(e * NumV + 0) * 4 + k] = vertices[_adjacency->getEdge(e, 0) + k];
		Ptr[(e * NumV + 0) * 4 + 3] = 1;
		//B
		for (int k = 0; k < 3; ++k)
			Ptr[(e * NumV + 1) * 4 + k] = vertices[_adjacency->getEdge(e, 1) + k];
		Ptr[(e * NumV + 1) * 4 + 3] = 1;
		//N
		Ptr[(e * NumV + 2) * 4 + 0] = float(_adjacency->getNofOpposite(e));
		for (int k = 1; k < 4; ++k)
			Ptr[(e * NumV + 2) * 4 + k] = 0;
		//Oi
		unsigned o = 0;
		for (; o < _adjacency->getNofOpposite(e); ++o)
		{
			for (int k = 0; k < 3; ++k)
				Ptr[(e * NumV + 2 + 1 + o) * 4 + k] = vertices[_adjacency->getOpposite(e, o) + k];
			Ptr[(e * NumV + 2 + 1 + o) * 4 + 3] = 1;
		}
		//zeros
		for (; o < _adjacency->getNofOpposite(e); ++o)
			for (int k = 0; k < 4; ++k)Ptr[(e * NumV + 2 + 1 + o) * 4 + k] = 0;
	}
	vbo->unmap();
}

void GSSV::createSidesVAO(vars::Vars& vars)
{
	FUNCTION_PROLOGUE("gssv", "model", "maxMultiplicity");

	Buffer* vbo = vars.get<Buffer>("gssv.sidesVBO");
	VertexArray* vao = vars.reCreate<VertexArray>("gssv.sidesVAO");
	unsigned int const NumV = getNofAttributes();

	for (unsigned a = 0; a < NumV; ++a)
	{
		vao->addAttrib(vbo, a, 4, GL_FLOAT, GLsizei(sizeof(float) * 4 * NumV), (GLintptr)(sizeof(float) * 4 * a));
	}
}

void GSSV::createSidesPrograms(vars::Vars& vars)
{
	FUNCTION_PROLOGUE("gssv", "maxMultiplicity", "args.gssv.useRefEdge", "args.gssv.cullSides", "args.gssv.useStencilExport");
	
	SGSSilTemplate TGS;

	TGS.Deterministic = true;
	TGS.ReferenceEdge = vars.getBool("args.gssv.useRefEdge");
	TGS.Universal = true;
	TGS.UseVS2GSArray = true;
	TGS.UseVertexArray = true;
	TGS.UseLayouts = true;
	TGS.UseStencilValueExport = vars.getBool("args.gssv.useStencilExport");
	TGS.CCWIsFrontFace = true;
	TGS.FrontFaceInside = false;
	TGS.CullSides = vars.getBool("args.gssv.cullSides");
	TGS.Visualize = false;
	TGS.FrontCapToInfinity = false;
	TGS.GenerateSides = true;
	TGS.GenerateCaps = false;
	TGS.Matrix = "mvp";
	TGS.MaxMultiplicity = vars.getUint32("maxMultiplicity");
	TGS.LightUniform = "LightPosition";
	TGS.VertexAttribName = "Position";

	std::shared_ptr<ge::gl::Shader>GSSilVer = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, genGsSilVertexShader(TGS));
	std::shared_ptr<ge::gl::Shader>GSSilGeom = std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER, genGsSilGeometryShader(TGS));
	std::shared_ptr<ge::gl::Shader>GSSilFrag = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, genGsSilFragmentShader(TGS));

	vars.reCreate<Program>("gssv.sidesProgram", GSSilVer, GSSilGeom, GSSilFrag);

    TGS.Visualize = true;
	vars.reCreate<Program>("gssv.sidesVisualizationProgram",
		std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, genGsSilVertexShader(TGS)),
		std::make_shared<ge::gl::Shader>(GL_GEOMETRY_SHADER, genGsSilGeometryShader(TGS)),
		std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, genGsSilFragmentShader(TGS)));
}

void GSSV::recomputeAdjacency(vars::Vars& vars)
{
	FUNCTION_PROLOGUE("gssv", "model", "maxMultiplicity");

	_adjacency = createAdjacency(vars);
}

void GSSV::drawSides(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	recomputeAdjacency(vars);
	createSidesVBO(vars);
	createSidesVAO(vars);
	createSidesPrograms(vars);
	
	const glm::mat4 mvp = projectionMatrix * viewMatrix;
	 
	vars.get<VertexArray>("gssv.sidesVAO")->bind();
	Program* program = vars.get<Program>("gssv.sidesProgram");

	program->use();
	program->setMatrix4fv("mvp", glm::value_ptr(mvp), 1, GL_FALSE);
	program->set4fv("LightPosition", glm::value_ptr(lightPosition), 1);

	glDrawArrays(GL_POINTS,0, GLsizei(_nofEdges));

	vars.get<VertexArray>("gssv.sidesVAO")->unbind();
}

void GSSV::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	recomputeAdjacency(vars);
	createCapsDrawer(vars);

	vars.get<GSCaps>("gssv.capsDrawer")->drawCaps(lightPosition, viewMatrix, projectionMatrix);
}

unsigned int GSSV::getNofAttributes() const
{
	return 2 + 1 + unsigned(_adjacency->getMaxMultiplicity());
}