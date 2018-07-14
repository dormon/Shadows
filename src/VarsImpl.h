#pragma once

#include <Vars.h>

class vars::VarsImpl {
 public:
  void* add(std::string const&    n,
            void*                 d,
            Destructor const&     dst,
            std::type_info const& t);

  void                  erase(std::string const& n);
  bool                  has(std::string const& n) const;
  std::type_info const& getType(std::string const& n) const;
  void checkTypes(std::string const& n, std::type_info const& t);
  void ifVarExistsThrow(std::string const&n)const;
  void ifVarDoesNotExistThrow(std::string const&n)const;
  ~VarsImpl();
  void* get(std::string const& n) const;

  template <typename T>
  T& add(std::string const& n, T const& v);
  template <typename T>
  T& get(std::string const& n) const;
  std::map<std::string, std::shared_ptr<Resource>> data;
  std::map<size_t, std::string>                    idToName;
  std::map<std::string, size_t>                    nameToId;
};

template <typename T>
T& vars::VarsImpl::add(std::string const& n, T const& v) {
  auto d   = new T(v);
  auto dst = getDestructor<T>();
  return reinterpret_cast<T&>(*reinterpret_cast<T*>(add(n, d, dst, typeid(T))));
}

template <typename T>
T& vars::VarsImpl::get(std::string const& n) const {
  return reinterpret_cast<T&>(*reinterpret_cast<T*>(get(n)));
}
