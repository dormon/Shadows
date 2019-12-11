#include <OdpmParams.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadOftsParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("odpm.args.resolution") = arg->getu32("--dpm-resolution", 512, "FTS irregular z-buffer resolution");
	vars.addUint32("odpm.args.depth") = arg->getu32("--dpm-depth", 10, "FTS irregular depth buffer max nof elements per texel");
	vars.addFloat("odpm.args.fovy") = arg->getf32("--dpm-fovy", 1.5707963267948966f, "FTS fovy");
	vars.addFloat("odpm.args.near") = arg->getf32("--dpm-near", 0.1f, "FTS near plane position");
	vars.addFloat("odpm.args.far") = arg->getf32("--dpm-far", 1000.f, "FTS far plane position");
	vars.addFloat("odpm.args.bias") = arg->getf32("--dpm-bias", 0.0000008f, "FTS bias when raytracing");
	vars.addBool("odpm.args.useFrusta") = arg->isPresent("--odpm-useFrusta", "Uses frustum test instead of raytracing");
	vars.addUint32("odpm.args.wgSize") = arg->getu32("--odpm-wgSize", 256, "OFTS frustum compute shader workgroup size");
}

