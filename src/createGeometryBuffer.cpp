#include <Vars/Vars.h>
#include <FunctionPrologue.h>
#include <glm/glm.hpp>
#include <Deferred.h>

void createGeometryBuffer(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","windowSize");

  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  vars.reCreate<GBuffer>("gBuffer",windowSize.x, windowSize.y);
}

