#include <Application.h>
#include <geGL/geGL.h>

namespace simple3DApp {

  Application::Application(int argc, char* argv[]) : argc(argc), argv(argv)
  {
    mainLoop = std::make_shared<sdl2cpp::MainLoop>();
    mainLoop->setIdleCallback(std::bind(&Application::draw, this));
    window = std::make_shared<sdl2cpp::Window>(512, 512);
    window->setEventCallback(SDL_MOUSEMOTION, [&](SDL_Event const& e) {
      mouseMove(e);
      return true;
    });
    window->setEventCallback(SDL_KEYDOWN, [&](SDL_Event const& e) {
      key(e, true);
      return true;
    });
    window->setEventCallback(SDL_KEYUP, [&](SDL_Event const& e) {
      key(e, false);
      return true;
    });
    window->createContext("rendering", 450u, sdl2cpp::Window::CORE,
                          sdl2cpp::Window::DEBUG);
    mainLoop->addWindow("primaryWindow", window);
    window->makeCurrent("rendering");
    ge::gl::init(SDL_GL_GetProcAddress);
    ge::gl::setHighDebugMessage();
  }

  void Application::start()
  {
    init();
    (*mainLoop)();
    deinit();
  }

  void Application::swap() { window->swap(); }

  void Application::mouseMove(SDL_Event const&) {}

  void Application::key(SDL_Event const&, bool) {}

  void Application::draw() {}

  void Application::init() {}

  void Application::deinit() {}

}  // namespace simple3DApp
