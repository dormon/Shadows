#include <FtsParams.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadFtsParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("args.fts.resolution") = arg->getu32("--fts-resolution", 512, "FTS irregular z-buffer resolution");
	vars.addUint32("args.fts.depth")      = arg->getu32("--fts-depth", 10, "FTS irregular depth buffer max nof elements per texel");
	vars.addFloat("args.fts.fovy")        = arg->getf32("--fts-fovy", 1.5707963267948966f, "FTS fovy");
	vars.addFloat("args.fts.near")        = arg->getf32("--fts-near", 0.1f, "FTS near plane position");
	vars.addFloat("args.fts.far")         = arg->getf32("--fts-far", 1000.f, "FTS far plane position");
	vars.addFloat("args.fts.bias")        = arg->getf32("--fts-bias", 0.0000008f, "FTS bias when raytracing");
}