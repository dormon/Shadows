#include <Vars/Vars.h>
#include <SDL2CPP/MainLoop.h>
#include <SDL2CPP/Window.h>

void stopAppAfterMaxFrame(vars::Vars&vars){
  if(vars.getSizeT("maxFrame") == 0)return;
  
  auto mainLoop = *vars.get<sdl2cpp::MainLoop*>("mainLoop");
  auto window   = *vars.get<sdl2cpp::Window  *>("window"  );
  if(vars.getSizeT("frameCounter") >= vars.getSizeT("maxFrame"))
    mainLoop->removeWindow(window->getId());
  vars.getSizeT("frameCounter")++;
}

