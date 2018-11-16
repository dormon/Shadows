#include <Methods.h>
#include <cassert>

size_t Methods::addMethodName(std::string const&name){
  size_t id = name2id.size();
  if(name2id.count(name) > 0)return name2id.at(name);
  name2id[name] = id  ;
  id2name[id  ] = name;
  return id;
}

std::string const& Methods::getName(size_t i)const{
  assert(id2name.count(i) > 0);
  return id2name.at(i);
}

size_t Methods::getId(std::string const&n)const{
  assert(name2id.count(n) > 0);
  return name2id.at(n);
}

size_t Methods::getNofMethods()const{
  return name2id.size();
}

bool Methods::hasMethod(std::string const&name){
  return name2id.count(name) > 0;
}

void Methods::createMethod(std::string const&name,vars::Vars&vars){
  if(!hasMethod(name)){
    vars.erase("shadowMethod");
    vars.getBool("useShadows") = false;
    vars.updateTicks("useShadows");
    return;
  }

  auto id = getId(name);
  constructors.at(id)(vars);
  vars.getBool("useShadows") = true;
}
