#include <Vars/Vars.h>

#include <Methods.h>
#include <imguiDormon/imgui.h>

void selectMethod(vars::Vars&vars){
  auto methods = vars.get<Methods>("methods");
  auto method = vars.getString("methodName");
  int oldMethodId;
  if(methods->hasMethod(method))
    oldMethodId = (int32_t)methods->getId(vars.getString("methodName"));
  else
    oldMethodId = (int32_t)methods->getNofMethods();
  int newMethodId = oldMethodId;

  std::vector<char const*>names;
  for(size_t i=0;i<methods->getNofMethods();++i)
    names.push_back(methods->getName(i).c_str());
  names.push_back("no shadow");
  
  ImGui::ListBox("method",&newMethodId,names.data(),(int32_t)names.size());
  if(newMethodId != oldMethodId){
    if(newMethodId < (int)methods->getNofMethods())
      vars.getString("methodName") = methods->getName(newMethodId);
    else
      vars.getString("methodName") = "no shadow";
    vars.updateTicks("methodName");
  }
}

