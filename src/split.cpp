#include <split.h>

std::vector<std::string>split(std::string txt,std::string const&splitter){
  std::vector<std::string>result;
  for(;;){
    auto where = txt.find(splitter);
    if(where == std::string::npos)break;
    result.push_back(txt.substr(0,where));
    txt = txt.substr(where+splitter.size());
  }
  result.push_back(txt);
  return result;
}
