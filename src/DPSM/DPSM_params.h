#pragma once

#include <memory>
#include <ArgumentViewer/Fwd.h>
#include <Vars/Fwd.h>

void loadDpsmParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg);