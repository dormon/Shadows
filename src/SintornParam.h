#pragma once

#include<cstdint>
#include<memory>
#include<ArgumentViewer/Fwd.h>
#include<Vars/Vars.h>

void loadSintornParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);
