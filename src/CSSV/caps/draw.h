#pragma once

#include<Vars/Fwd.h>
#include<glm/glm.hpp>

namespace cssv{
void drawCaps(vars::Vars&vars,
    glm::vec4 const&lightPosition   ,
    glm::mat4 const&viewMatrix      ,
    glm::mat4 const&projectionMatrix);
}
