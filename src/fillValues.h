#pragma once

#include<sstream>
#include<split.h>

template<typename... ARGS>
std::string fillValues(std::string const&a,std::string placeholder = "%%",ARGS const&...args){
  std::vector<std::string>svalues = {toStr(args)...};
  auto orig = split(a,placeholder);
  std::stringstream ss;
  for(size_t i=0;i<orig.size();++i){
    ss << orig[i];
    if(i < svalues.size())ss << svalues[i];
  }
  return ss.str();
}
