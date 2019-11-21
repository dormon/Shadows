#include <DpsvParams.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadDpsvParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addBool("dpsv.args.cullFront") = arg->isPresent("--dpsv-cullFront", "Enables front face culling for dpsv");
	vars.addUint32("dpsv.args.wgSize") = arg->getu32("--dpsv-wgSize", 512, "Workgroup size");
	vars.addFloat("dpsv.args.bias") = arg->getf32("--dpsv-bias", 0.0001f, "DPSV bias used during build");
	vars.addUint32("dpsv.args.numWg") = arg->getu32("--dpsv-numWg", 32, "DPSV nof work groups for compute shader persistent threads");
	vars.addBool("dpsv.args.useDepthOptimization") = arg->isPresent("--dpsv-depthOptim", "Enabled depth optimization");
	vars.addUint32("dpsv.args.algVersion") = arg->getu32("--dpsv-algVer", 0, "DPSV algorithm version. 0 - stack, 1 - stackless, 2 - hybrid");
}

