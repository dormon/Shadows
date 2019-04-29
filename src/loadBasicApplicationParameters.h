#pragma once

#include<memory>

#include<Vars/Fwd.h>
#include<ArgumentViewer/Fwd.h>

void loadBasicApplicationParameters(vars::Vars&vars,std::shared_ptr<argumentViewer::ArgumentViewer>const&args);
