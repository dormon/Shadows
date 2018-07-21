#include <Function.h>
#include <FunctionImpl.h>
#include <sstream>
#include <stdexcept>

Function::Function(std::vector<std::type_info>const& inputTypes ,
                   std::vector<std::type_info>const& outputTypes,
                   vars::Vars                      & vars       )
{
  impl = std::make_unique<FunctionImpl>(inputTypes, outputTypes, vars);
}

Function::~Function(){}

void Function::unbindInput(size_t id) { impl->unbindInput(id); }

void Function::bindInput(size_t id, std::string const& v)
{
  impl->bindInput(id, v);
}

void Function::unbindOutput(size_t id) { impl->unbindOutput(id); }

void Function::bindOutput(size_t id, std::string const& v)
{
  impl->bindOutput(id, v);
}

void Function::operator()(){
  impl->operator()();
}
