#include <TSSV_Params.h>
#include <ArgumentViewer/ArgumentViewer.h>

void loadTSSVParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& args)
{
	vars.addBool("args.tssv.useRefEdge")       = args->isPresent("--tssv-useRefEdge", "Use Reference Edge");
	vars.addBool("args.tssv.useStencilExport") = args->isPresent("--tssv-useStencilExport", "Use stencil value export. AMD ONLY!");
	vars.addBool("args.tssv.cullSides")        = args->isPresent("--tssv-cullSides", "Cull Sides");
}
