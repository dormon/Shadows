#pragma once

#include<iostream>
#include<Vars/Vars.h>
#include<vector>
#include<array>
#include<map>
#include<array>

struct UniquePinData{
  size_t cardinality = 0;
  size_t seenTicks = 0;
};

using SharedResource = std::shared_ptr<vars::Resource>;

struct PinData{
  PinData(std::type_info const&t):type(t){}
  std::type_info const&type;
  SharedResource data;
};

class FunctionImpl{
  public:
    FunctionImpl(
        std::vector<std::type_info>const&inputTypes ,
        std::vector<std::type_info>const&outputTypes,
        vars::Vars&vars);
    ~FunctionImpl();
    enum PinType{
      INPUT  = 0,
      OUTPUT = 1,
    };
    void bindInput   (size_t id,std::string const&v);
    void unbindInput (size_t id);
    void bindOutput  (size_t id,std::string const&v);
    void unbindOutput(size_t id);
    std::string getPinTypeName                 (PinType const&type)const;
    void ifPinTypeDoesNotMatchResourceTypeThrow(PinType const&type,size_t id,std::string const&n)const;
    void ifResourceDoesNotExistThrow           (PinType const&type,size_t id,std::string const&n)const;
    void decreaseUniquePinDataCardinality      (PinType const&type,SharedResource const&r);
    void increaseUniquePinDataCardinality      (PinType const&type,SharedResource const&r);
    void ifUniquePinDataDoesNotExistCreateIt   (PinType const&type,SharedResource const&r);
    void ifPinDataCardinalityIsZeroEraseIt     (PinType const&type,SharedResource const&r);
    void operator()();
    SharedResource getResourceFromVarsOrThrow(std::string const&n);
    vars::Vars&vars;
    void unbindPin(PinType const&type,size_t id);
    void bindPin  (PinType const&type,size_t id,std::string const&v);
    using Pins = std::vector<PinData>;
    using UniqueData = std::map<SharedResource,UniquePinData>;
    std::array<Pins      ,2>pins      ;
    std::array<UniqueData,2>uniqueData;
};
