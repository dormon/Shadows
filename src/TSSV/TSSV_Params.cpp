#include <TSSV_Params.h>
#include <ArgumentViewer/ArgumentViewer.h>
#include <Vars/Vars.h>

void loadTSSVParams(vars::Vars& vars, std::shared_ptr<argumentViewer::ArgumentViewer>const& args)
{
	vars.addBool("tssv.args.useRefEdge") = args->isPresent("--tssv-useRefEdge", "Use Reference Edge");
	vars.addBool("tssv.args.useStencilExport") = args->isPresent("--tssv-useStencilExport", "Use stencil value export. AMD ONLY!");
	vars.addBool("tssv.args.cullSides") = args->isPresent("--tssv-cullSides", "Cull Sides");
}
