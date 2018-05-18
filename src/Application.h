#pragma once

#include<SDL2CPP/Window.h>

namespace simple3DApp{

class Application{
  public:
    Application(int argc,char*argv[]);
    void start();
    void swap();
    virtual void mouseMove(SDL_Event const&e);
    virtual void key(SDL_Event const&e,bool down);
    virtual void draw();
    virtual void init();
    virtual void deinit();
  protected:
    std::shared_ptr<sdl2cpp::MainLoop>mainLoop;
    std::shared_ptr<sdl2cpp::Window>window;
    int argc;
    char**argv;
};

}
