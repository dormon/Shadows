#pragma once

#include<cstdint>
#include<memory>
#include<ArgumentViewer/Fwd.h>
#include<Vars.h>

void loadCubeShadowMappingParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);

