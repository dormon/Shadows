#include <GSSV_params.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadGSSVParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& args)
{
	vars.addBool("args.gssv.useRefEdge")		= args->isPresent("--gssv-useRefEdge", "Use Reference Edge");
	vars.addBool("args.gssv.useStencilExport")	= args->isPresent("--gssv-useStencilExport", "Use stencil value export. AMD ONLY!");
	vars.addBool("args.gssv.cullSides")			= args->isPresent("--gssv-cullSides", "Cull Sides");
}
