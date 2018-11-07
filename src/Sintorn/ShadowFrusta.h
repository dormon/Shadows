#pragma once

#include<glm/glm.hpp>
#include<Vars/Vars.h>

void computeShadowFrusta(vars::Vars&vars,glm::vec4 const&lightPosition,glm::mat4 mvp);
