#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>

namespace vars {

using Destructor = std::function<void(void*)>;
using OnDestroy  = std::function<void(void*)>;
using OnChange   = std::function<void(void*, void*)>;

class VarsImpl;
class Vars;

template<typename T>
inline Destructor getDestructor();

}  // namespace vars

class vars::Vars {
 public:
  Vars();
  ~Vars();
  void*        add(std::string const&    n,
                   void*                 d,
                   Destructor const&     dst,
                   std::type_info const& t);
  bool&        addBool(std::string const& n, bool v = false);
  std::string& addString(std::string const& n, std::string const& v = "");
  size_t&      addSizeT(std::string const& n, size_t v = 0);
  float&       addFloat(std::string const& n, float v = 0.f);
  uint32_t&    addUint32(std::string const& n, uint32_t v = 0);
  void*        get(std::string const& n) const;
  std::string& getString(std::string const& n) const;
  bool&        getBool(std::string const& n) const;
  size_t&      getSizeT(std::string const& n) const;
  float&       getFloat(std::string const& n) const;
  uint32_t&    getUint32(std::string const& n) const;
  void         erase(std::string const& n);
  bool         has(std::string const& n) const;
  std::type_info const& getType(std::string const& n) const;
  template <typename CLASS, typename... ARGS>
  CLASS* add(std::string const& n, ARGS const&... args);
  template <typename CLASS>
  CLASS* get(std::string const& n) const;

 private:
  void checkTypes(std::string const& n, std::type_info const& t) const;
  friend class VarsImpl;
  std::unique_ptr<VarsImpl> impl;
};

template <typename CLASS, typename... ARGS>
CLASS* vars::Vars::add(std::string const& n, ARGS const&... args) {
  void* data = new CLASS{args...};
  auto r = add(n, data, getDestructor<CLASS>(), typeid(CLASS));
  return reinterpret_cast<CLASS*>(r);
}

template <typename CLASS>
CLASS* vars::Vars::get(std::string const& n) const {
  checkTypes(n, typeid(CLASS));
  return reinterpret_cast<CLASS*>(get(n));
}

template<typename T>
inline vars::Destructor vars::getDestructor(){
  return [](void* ptr) { delete reinterpret_cast<T*>(ptr);};
}
