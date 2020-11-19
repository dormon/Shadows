#pragma once

#include<memory>
#include<ArgumentViewer/Fwd.h>
#include<Vars/Fwd.h>

namespace cssv{

void loadParams(
    vars::Vars&vars,
    std::shared_ptr<argumentViewer::ArgumentViewer>const&arg);

}
