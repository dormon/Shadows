#pragma once

#include <iostream>
#include <Vars/Fwd.h>

namespace rssv{
std::string getConfigShader     (vars::Vars&vars);
std::string getDebugConfigShader(vars::Vars&vars);
}
