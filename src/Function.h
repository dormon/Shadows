#pragma once

#include <Vars/Fwd.h>
#include <vector>

class FunctionImpl;
class Function{
  public:
    Function(
        std::vector<std::type_info>const&inputTypes ,
        std::vector<std::type_info>const&outputTypes,
        vars::Vars                    &vars       );
    virtual ~Function();
    void operator()();
    void bindInput   (size_t id,std::string const&v);
    void unbindInput (size_t id);
    void bindOutput  (size_t id,std::string const&v);
    void unbindOutput(size_t id);
  protected:
    friend class FunctionImpl;
    std::unique_ptr<FunctionImpl>impl;
};

