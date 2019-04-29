#include<FunctionPrologue.h>
#include<Methods.h>

void createMethod(vars::Vars&vars){
  FUNCTION_PROLOGUE("all","methodName");

  auto const methodName = vars.getString("methodName"); 
  auto methods = vars.get<Methods>("methods");
  methods->createMethod(methodName,vars);
}

