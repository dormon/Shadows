#include <Vars.h>

using namespace vars;

class Element {
 public:
  Element(void* d, Destructor const& dst, std::type_info const& t)
      : data(d), destructor(dst), type(t) {}
  ~Element() { destructor(data); }
  void*                 data;
  Destructor            destructor;
  std::type_info const& type;
};

class vars::VarsImpl {
 public:
  std::map<std::string, Element> data;
  void*                          add(std::string const&    n,
                                     void*                 d,
                                     Destructor const&     dst,
                                     std::type_info const& t) {
    if (has(n))
      throw std::runtime_error(std::string("variable: ") + n +
                               " already exists!");
    data.emplace(std::piecewise_construct, std::forward_as_tuple(n),
                 std::forward_as_tuple(d, dst, t));
    return d;
  }
  void* get(std::string const& n) const {
    if (!has(n))
      throw std::runtime_error(std::string("variable: ") + n +
                               " does not exist!");
    return data.at(n).data;
  }
  template<typename T>
  T&add(std::string const&n,T const&v){
    auto d = new T(v);
    auto dst = getDestructor<T>();
    return reinterpret_cast<T&>(*reinterpret_cast<T*>(add(n,d,dst,typeid(T))));
  }
  template<typename T>
  T&get(std::string const&n)const{
    return reinterpret_cast<T&>(*reinterpret_cast<T*>(get(n)));
  }
  void erase(std::string const& n) { data.erase(n); }
  bool has(std::string const& n) const { return data.count(n) > 0; }
  std::type_info const& getType(std::string const& n) const {
    return data.at(n).type;
  }
  void checkTypes(std::string const& n, std::type_info const& t) {
    if (getType(n) == t) return;
    throw std::runtime_error(std::string("variable: ") + n +
                             " has different type");
  }
  ~VarsImpl() {}
};

void*Vars::add(std::string const&                n,
               void*                             d,
               std::function<void(void*)> const& dst,
               std::type_info const&             t) {
  return impl->add(n, d, dst, t);
}

bool& Vars::addBool(std::string const&n,bool v){
  return impl->add<bool>(n,v);
}

std::string& Vars::addString(std::string const&n,std::string const&v){
  return impl->add<std::string>(n,v);
}

size_t&Vars::addSizeT(std::string const&n,size_t v){
  return impl->add<size_t>(n,v);
}

float&Vars::addFloat(std::string const& n, float v){
  return impl->add<float>(n,v);
}

uint32_t&Vars::addUint32(std::string const& n, uint32_t v){
  return impl->add<uint32_t>(n,v);
}

void* Vars::get(std::string const& n) const { return impl->get(n); }

std::string&Vars::getString(std::string const&n)const{
  return impl->get<std::string>(n);
}

bool&Vars::getBool(std::string const&n)const{
  return impl->get<bool>(n);
}

size_t&Vars::getSizeT(std::string const&n)const{
  return impl->get<size_t>(n);
}

float&Vars::getFloat(std::string const& n) const{
  return impl->get<float>(n);
}

uint32_t&Vars::getUint32(std::string const& n) const{
  return impl->get<uint32_t>(n);
}

void Vars::erase(std::string const& n) { impl->erase(n); }

bool Vars::has(std::string const& n) const { return impl->has(n); }

std::type_info const& Vars::getType(std::string const& n) const {
  return impl->getType(n);
}

void Vars::checkTypes(std::string const& n, std::type_info const& t) const {
  impl->checkTypes(n, t);
}

Vars::Vars() { impl = std::make_unique<VarsImpl>(); }

Vars::~Vars() {}
