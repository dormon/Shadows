#pragma once

#include <memory>
#include <ArgumentViewer/Fwd.h>
#include <Vars/Vars.h>

void loadDpsvParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg);