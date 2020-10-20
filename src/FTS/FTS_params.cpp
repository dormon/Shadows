#include <FTS_params.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadFtsParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("fts.args.resolution") = arg->getu32("--fts-res", 1024, "Resolution (both X and Y)");
	vars.addUint32("fts.args.wgSize")     = arg->getu32("--fts-wgSize", 128, "Workgroup size");
	vars.addFloat("fts.args.nearZ")       = arg->getf32("--fts-nearZ", 0.1f, "Near clipping plane");
	vars.addFloat("fts.args.farZ")        = arg->getf32("--fts-farZ", 1000.f, "Far cliping plane");
	vars.addFloat("fts.args.fovY")        = arg->getf32("--fts-fovY", 1.5707963267948966f, "FovY");
}