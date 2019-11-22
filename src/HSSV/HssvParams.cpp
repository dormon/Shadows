#include <HssvParams.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadHssvParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("hssv.args.octreeDepth") = arg->getu32("--hssv-octreeDepth", 5, "Octree depth");
	vars.addFloat ("hssv.args.sceneScale") = arg->getf32("--hssv-sceneScale", 10.f, "Scaling factor of scene's AABB to form an octree");
	vars.addUint32("hssv.args.wgSize") = arg->getu32("--hssv-wgSize", 256, "HSSV worgroup size");
	vars.addBool  ("hssv.args.forceBuild") = arg->isPresent("--hssv-forceBBuild", "Build the octree even though there is a serialized version");
	vars.addBool  ("hssv.args.drawFromCpu") = arg->isPresent("--hssv-drawFromCPU", "Use CPU traversal rather than GPU");
	vars.addBool  ("hssv.args.buildOnCpu") = arg->isPresent("--hssv-buildOnCpu", "Use CPU traversal rather than GPU");
	vars.addBool  ("hssv.args.noSaveFile") = arg->isPresent("--hssv-noSaveFile", "Will not save generated octree to a file");
	vars.addBool  ("hssv.args.compressionLevel") = arg->isPresent("--hssv-noCompression", "Turns off octree compression");
}
