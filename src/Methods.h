#pragma once

#include <iostream>
#include <map>
#include <Vars/Vars.h>

class Methods{
  public:
    template<typename T>
    size_t add(std::string const&name);
    std::string const& getName(size_t i)const;
    size_t getId(std::string const&n)const;
    size_t getNofMethods()const;
    void createMethod(std::string const&name,vars::Vars&vars);
    bool hasMethod(std::string const&name);
  protected:
    size_t addMethodName(std::string const&name);
    std::map<std::string,size_t>name2id;
    std::map<size_t,std::string>id2name;
    std::map<size_t,std::function<void(vars::Vars&)>>constructors;
};

template<typename T>
size_t Methods::add(std::string const&name){
  auto const id = addMethodName(name);
  constructors[id] = [](vars::Vars&vars){vars.reCreate<T>("shadowMethod",vars);};
}
