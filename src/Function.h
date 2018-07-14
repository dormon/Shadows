#pragma once

#include <Vars.h>

/*
struct UniqueInputData{
  vars::Element*element;
  size_t cardinality = 0;
  size_t ticks = 0;
};

struct Input{
  std::type_info const&type;
};

class Function{
  public:
    Function(
        std::vector<std::type_info const&>inputTypes,
        std::vector<std::type_info const&>outputTypes,
        vars::Vars&vars);
    virtual ~Function();
    void operator()();
    void bindInput (size_t id,std::string const&v);
    void bindOutput(size_t id,std::string const&v);
  protected:
    virtual void execute() = 0;
    vars::Vars&vars;
    std::vector<vars::Element*>inputs;
    std::vector<vars::Element*>outputs;
    std::map<vars::Element*,size_t>uniqueInputs;
    std::map<vars::Element*,size_t>uniqueOutputs;
};

*/
