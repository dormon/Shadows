#include <OFTS_params.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadOftsParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("dpsv.args.res") = arg->getu32("--ofts-res", 1024, "Resolution (both X and Y)");
}