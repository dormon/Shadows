#include <DpmParams.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadDpmParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("dpm.args.resolution") = arg->getu32("--dpm-resolution", 512, "DPM irregular z-buffer resolution");
	vars.addUint32("dpm.args.depth") = arg->getu32("--dpm-depth", 10, "DPM irregular depth buffer max nof elements per texel");
	vars.addFloat("dpm.args.fovy") = arg->getf32("--dpm-fovy", 1.5707963267948966f, "DPM fovy");
	vars.addFloat("dpm.args.near") = arg->getf32("--dpm-near", 0.1f, "DPM near plane position");
	vars.addFloat("dpm.args.far") = arg->getf32("--dpm-far", 1000.f, "DPM far plane position");
	vars.addFloat("dpm.args.bias") = arg->getf32("--dpm-bias", 0.0000008f, "DPM bias when raytracing");
}