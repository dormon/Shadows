#include <FTS_params.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadFtsParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("fts.args.resolution") = arg->getu32("--fts-res", 1024, "Resolution (both X and Y)");
	vars.addUint32("fts.args.wgSize")     = arg->getu32("--fts-wgSize", 64, "Workgroup size");
	vars.addFloat("fts.args.nearZ")       = arg->getf32("--fts-nearZ", 1.f, "Near clipping plane");
	vars.addFloat("fts.args.farZ")        = arg->getf32("--fts-farZ", 1000.f, "Far cliping plane");
	vars.addFloat("fts.args.fovY")        = arg->getf32("--fts-fovY", 1.5707963267948966f, "FovY");
	vars.addFloat("fts.args.traversalBias") = arg->getf32("--fts-bias", 0.001f, "Bias when creating shadow frusta");
	vars.addUint32("fts.args.longListTreshold") = arg->getu32("--fts-treshold", 1024, "Treshold for reprojecting long lists");
	vars.addUint32("fts.args.heatmapRes") = arg->getu32("--fts-heatmapRes", 512, "Resolution of heatmap texture");
}