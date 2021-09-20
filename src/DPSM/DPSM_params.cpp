#include "DPSM_params.h"
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadDpsmParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("dpsm.args.resolution") = arg->getu32("--dpsm-res", 1024, "Resolution (both X and Y)");
	vars.addFloat("dpsm.args.near")      = arg->getf32("--dpsm-near", 0.1f, "Paraboloid near clipping plane");
	vars.addFloat("dpsm.args.far")       = arg->getf32("--dpsm-far", 100.f, "Paraboloid far clipping plane");
}