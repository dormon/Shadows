#pragma once

#include<ArgumentViewer/Fwd.h>
#include<Vars/Fwd.h>
#include<memory>

void loadVSSVParams(
    vars::Vars &vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);

