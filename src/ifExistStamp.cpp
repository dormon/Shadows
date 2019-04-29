#include <Vars/Vars.h>
#include <TimeStamp.h>

void ifExistStamp(vars::Vars&vars,std::string const&n){
  if (vars.has("timeStamp")) vars.get<TimeStamp>("timeStamp")->stamp(n);
}

