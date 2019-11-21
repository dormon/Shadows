#include <HSSV.h>
#include <GSCaps.h>

#include <FunctionPrologue.h>

HSSV::HSSV(vars::Vars& vars) : ShadowVolumes(vars)
{
}

HSSV::~HSSV()
{
	vars.erase("hssv.objects");
}

void HSSV::drawCaps(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	createCapsDrawer();

	vars.get<GSCaps>("hssv.objects.capsDrawer")->drawCaps(lightPosition, viewMatrix, projectionMatrix);
}

void HSSV::drawUser(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{

}

void HSSV::createCapsDrawer()
{
	FUNCTION_PROLOGUE("hssv.objects", "renderModel", "maxMultiplicity");

	vars.reCreate<GSCaps>("hssv.objects.capsDrawer", vars);
}

