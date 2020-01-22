#pragma once

#include <Vars/Fwd.h>

namespace rssv::debug{
void dumpBasic(vars::Vars&vars);
void dumpSamples(vars::Vars&vars);
void dumpNodePool(vars::Vars&vars);
void dumpAABBPool(vars::Vars&vars);
void dumpData(vars::Vars&vars);
void dumpSilhouettes(vars::Vars&vars);
void dumpTraverse(vars::Vars&vars);
}
