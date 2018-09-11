#pragma once

#include<cstdint>
#include<memory>
#include<glm/glm.hpp>
#include<ArgumentViewer/Fwd.h>
#include<Vars/Vars.h>

namespace rssv{
void loadParams(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);
}
