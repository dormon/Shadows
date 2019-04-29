#include<Vars/Vars.h>
#include<TimeStamp.h>

void ifExistBeginStamp(vars::Vars&vars){
  if (vars.has("timeStamp")) vars.get<TimeStamp>("timeStamp")->begin();
}

