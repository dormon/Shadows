#include <HssvParams.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadHssvParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("hssv.args.octreeDepth") = arg->getu32("--hssv-octreeDepth", 5, "Octree depth");
	vars.addFloat ("hssv.args.sceneScale") = arg->getf32("--hssv-sceneScale", 10.f, "Scaling factor of scene's AABB to form an octree");
	vars.addUint32("hssv.args.wgSize") = arg->getu32("--hssv-wgSize", 1024, "HSSV worgroup size for side CS");
	vars.addBool  ("hssv.args.forceBuild") = arg->isPresent("--hssv-forceBuild", "Build the octree even though there is a serialized version");
	vars.addBool  ("hssv.args.drawCpu") = arg->isPresent("--hssv-drawCpu", "Use CPU traversal rather than GPU");
	vars.addBool  ("hssv.args.buildCpu") = arg->isPresent("--hssv-buildCpu", "Use CPU traversal rather than GPU");
	vars.addBool  ("hssv.args.dontStoreOctree") = arg->isPresent("--hssv-dontStoreOctree", "Will not save generated octree to a file");
	vars.addBool  ("hssv.args.noCompression") = arg->isPresent("--hssv-noCompression", "Turns off octree compression");
	vars.addBool  ("hssv.args.useExperimental") = arg->isPresent("--hssv-useExperimental", "Use experimental version of traversal");
}
