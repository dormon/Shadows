#pragma once

#include <functional>

namespace vars {
class Resource;
using Destructor = std::function<void(void*)>;
using OnChange   = std::function<void()>;
};  // namespace vars

class vars::Resource {
 public:
  Resource(void* d, Destructor const& dst, std::type_info const& t);
  ~Resource();
  void                  updateTicks();
  size_t                getTicks() const;
  std::type_info const& getType() const;
  void                  setChangeCallback(OnChange const& clb);
  void*                 getData() const;

 private:
  void*                 data;
  size_t                ticks          = 1;
  OnChange              changeCallback = nullptr;
  Destructor            destructor;
  std::type_info const& type;
};
