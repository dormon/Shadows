#include <sstream>

#include <getMethodNameList.h>
#include <FunctionPrologue.h>
#include <Methods.h>

std::string getMethodNameList(vars::Vars&vars){
  vars::Caller caller(vars,__FUNCTION__);
  std::stringstream ss;
  auto const methods = vars.get<Methods>("methods");
  bool first = true;
  for(size_t i=0;i<methods->getNofMethods();++i){
    if(first)first = false;
    else ss << "/";
    ss << methods->getName(i);
  }
  return ss.str();
}
