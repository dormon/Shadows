#pragma once

#include<cstdint>
#include<memory>
#include<glm/glm.hpp>
#include<ArgumentViewer/Fwd.h>
#include<Vars/Vars.h>

void loadRSSVParams(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);
