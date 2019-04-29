#include <Vars/Vars.h>
#include <TimeStamp.h>

void ifExistEndStamp(vars::Vars&vars,std::string const&n){
  if (vars.has("timeStamp")) vars.get<TimeStamp>("timeStamp")->end(n);
}

