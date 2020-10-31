#include <OdpmParams.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadOdpmParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& arg)
{
	vars.addUint32("odpm.args.wgSize")     = arg->getu32("--odpm-wgSize", 128, "ODPM Compute shader WG Size");
	vars.addUint32("odpm.args.resolution") = arg->getu32("--odpm-resolution", 512, "ODPM irregular z-buffer resolution");
	vars.addUint32("odpm.args.depth")      = arg->getu32("--odpm-depth", 10, "ODPM irregular depth buffer max nof elements per texel");
	vars.addFloat("odpm.args.near")        = arg->getf32("--odpm-near", 0.1f, "ODPM near plane position");
	vars.addFloat("odpm.args.far")         = arg->getf32("--odpm-far", 1000.f, "ODPM far plane position");
	vars.addBool("odpm.args.useFrusta")    = arg->isPresent("--odpm-useFrusta", "Use frustum test instead of ray tracing");
	vars.addFloat("odpm.args.bias")        = arg->getf32("--odpm-bias", 0.0000008f, "ODPM bias when raytracing");
}
