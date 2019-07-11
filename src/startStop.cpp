#include <startStop.h>
#include <Vars/Vars.h>
#include <chrono>

void start(vars::Vars&vars,std::string const&n){
  auto v = vars.addOrGet<std::chrono::time_point<std::chrono::high_resolution_clock>>(n);
  *v = std::chrono::high_resolution_clock::now();
}

void stop(vars::Vars&vars,std::string const&n){
  auto const newTime = std::chrono::high_resolution_clock::now();
  auto v = vars.get<std::chrono::time_point<std::chrono::high_resolution_clock>>(n);
  std::chrono::duration<float>const elapsed = newTime - *v;
  auto const dt = elapsed.count();
  vars.addOrGetFloat(n+"_time") = dt;
}

