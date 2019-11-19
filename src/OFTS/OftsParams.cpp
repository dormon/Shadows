#include <OftsParams.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadOftsParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("args.ofts.resolution") = arg->getu32("--fts-resolution", 512, "FTS irregular z-buffer resolution");
	vars.addUint32("args.ofts.depth")      = arg->getu32("--fts-depth", 10, "FTS irregular depth buffer max nof elements per texel");
	vars.addFloat ("args.ofts.fovy")       = arg->getf32("--fts-fovy", 1.5707963267948966f, "FTS fovy");
	vars.addFloat ("args.ofts.near")       = arg->getf32("--fts-near", 0.1f, "FTS near plane position");
	vars.addFloat ("args.ofts.far")        = arg->getf32("--fts-far", 1000.f, "FTS far plane position");
	vars.addFloat ("args.ofts.bias")       = arg->getf32("--fts-bias", 0.0000008f, "FTS bias when raytracing");
	vars.addBool  ("args.ofts.useFrusta")  = arg->isPresent("--ofts-useFrusta", "Uses frustum test instead of raytracing");
	vars.addUint32("args.ofts.wgSize")     = arg->getu32("--ofts-wgSize", 256, "OFTS frustum compute shader workgroup size");
}

