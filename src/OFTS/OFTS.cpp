#include <OFTS.h>

#include <Vars/Vars.h>

OFTS::~OFTS()
{
	vars.erase("ofts.objects");
}

OFTS::OFTS(vars::Vars& vars) : ShadowMethod(vars)
{
	IsValid = IsConservativeRasterizationSupported();

}

void OFTS::create(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{

}