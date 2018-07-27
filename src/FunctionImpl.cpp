#include<FunctionImpl.h>
#include<Vars/Resource.h>
#include<sstream>

FunctionImpl::FunctionImpl(
    std::vector<std::type_info>const& inputTypes,
    std::vector<std::type_info>const& outputTypes,
    vars::Vars                           & vars)
    : vars(vars)
{
  for (auto const& i : inputTypes) pins[INPUT].emplace_back(i);
  for (auto const& o : outputTypes) pins[OUTPUT].emplace_back(o);
}

FunctionImpl::~FunctionImpl() {}

void FunctionImpl::operator()() { }

std::string FunctionImpl::getPinTypeName(PinType const& type) const
{
  std::string const pinTypeNames[] = {"input", "output"};
  return pinTypeNames[type];
}

void FunctionImpl::ifResourceDoesNotExistThrow(PinType const&     type,
                                           size_t             id,
                                           std::string const& n) const
{
  if (vars.has(n)) return;
  std::stringstream ss;
  ss << "Cannot bind variable: \"" << n << "\" to " << getPinTypeName(type)
     << ": " << id << " - variable does not exist";
  throw std::runtime_error(ss.str());
}

void FunctionImpl::ifPinTypeDoesNotMatchResourceTypeThrow(
    PinType const&     type,
    size_t             id,
    std::string const& n) const
{
  auto const& varType = vars.getType(n);
  auto const& pinType = pins[type].at(id).type;
  if (pinType == varType) return;
  std::stringstream ss;
  ss << "Cannot bind variable: \"" << n << "\" with type: " << varType.name()
     << " to " << getPinTypeName(type) << ": " << id
     << " with type: " << pinType.name() << " - different types";
  throw std::runtime_error(ss.str());
}

void FunctionImpl::decreaseUniquePinDataCardinality(
    PinType const&                         type,
    std::shared_ptr<vars::Resource> const& r)
{
  uniqueData[type].at(r).cardinality--;
}

void FunctionImpl::increaseUniquePinDataCardinality(
    PinType const&                         type,
    std::shared_ptr<vars::Resource> const& r)
{
  uniqueData[type].at(r).cardinality++;
}

void FunctionImpl::ifUniquePinDataDoesNotExistCreateIt(
    PinType const&                         type,
    std::shared_ptr<vars::Resource> const& r)
{
  if (uniqueData[type].count(r) > 0) return;
  uniqueData[type][r] = {0, r->getTicks() - 1};
}

void FunctionImpl::ifPinDataCardinalityIsZeroEraseIt(
    PinType const&                         type,
    std::shared_ptr<vars::Resource> const& r)
{
  if (uniqueData[type].at(r).cardinality == 0) uniqueData[type].erase(r);
}

void FunctionImpl::unbindPin(PinType const& type, size_t id)
{
  if (!pins[type].at(id).data) return;
  auto resource          = pins[type].at(id).data;
  pins[type].at(id).data = nullptr;
  decreaseUniquePinDataCardinality(type, resource);
  ifPinDataCardinalityIsZeroEraseIt(type, resource);
}

void FunctionImpl::bindPin(PinType const& type, size_t id, std::string const& v)
{
  ifResourceDoesNotExistThrow(type, id, v);
  ifPinTypeDoesNotMatchResourceTypeThrow(type, id, v);
  unbindPin(type,id);
  auto resource          = vars.getResource(v);
  pins[type].at(id).data = resource;
  ifUniquePinDataDoesNotExistCreateIt(type, resource);
  increaseUniquePinDataCardinality(type, resource);
}

void FunctionImpl::unbindInput(size_t id) { unbindPin(INPUT, id); }

void FunctionImpl::bindInput(size_t id, std::string const& v)
{
  bindPin(INPUT, id, v);
}

void FunctionImpl::unbindOutput(size_t id) { unbindPin(OUTPUT, id); }

void FunctionImpl::bindOutput(size_t id, std::string const& v)
{
  bindPin(OUTPUT, id, v);
}

