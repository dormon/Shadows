#include <GSSV_params.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadGSSVParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& args)
{
	vars.addBool("gssv.args.useRefEdge") = args->isPresent("--gssv-useRefEdge", "Use Reference Edge");
	vars.addBool("gssv.args.useStencilExport") = args->isPresent("--gssv-useStencilExport", "Use stencil value export. AMD ONLY!");
	vars.addBool("gssv.args.cullSides") = args->isPresent("--gssv-cullSides", "Cull Sides");
}
