#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Vars/Vars.h>
#include <geGL/geGL.h>
#include <geGL/StaticCalls.h>
#include <imguiDormon/imgui.h>

#include <Deferred.h>
#include <FunctionPrologue.h>
#include <divRoundUp.h>

#include <RSSV/mortonShader.h>
#include <RSSV/debug/drawDebug.h>
#include <RSSV/debug/dumpData.h>
#include <RSSV/debug/dumpSamples.h>
#include <RSSV/debug/drawSamples.h>
#include <RSSV/debug/drawNodePool.h>
#include <RSSV/debug/drawTraverse.h>
#include <RSSV/debug/drawSF.h>
#include <RSSV/debug/drawEdges.h>
#include <RSSV/debug/drawSilhouettes.h>
#include <RSSV/debug/drawEdgePlanes.h>
#include <RSSV/debug/drawSVSides.h>
#include <RSSV/debug/drawBridges.h>
#include <RSSV/debug/drawStencil.h>
#include <RSSV/configShader.h>
#include <RSSV/config.h>

using namespace ge::gl;
using namespace std;

namespace rssv::debug{

void prepareCommon(vars::Vars&vars){
  FUNCTION_PROLOGUE("rssv.method.debug");
  vars.reCreate<VertexArray>("rssv.method.debug.vao");
}

void blitDepth(vars::Vars&vars){
  auto windowSize = *vars.get<glm::uvec2>("windowSize");
  auto gBuffer = vars.get<GBuffer>("gBuffer");
  glBlitNamedFramebuffer(
      gBuffer->fbo->getId(),
      0,
      0,0,windowSize.x,windowSize.y,
      0,0,windowSize.x,windowSize.y,
      GL_DEPTH_BUFFER_BIT,
      GL_NEAREST);
}

}

void rssv::drawDebug(vars::Vars&vars){
  debug::prepareCommon(vars);
  debug::blitDepth(vars);

  auto&levelsToDraw       = vars.addOrGetUint32("rssv.method.debug.levelsToDraw",0);
  auto&drawTightAABB      = vars.addOrGetBool  ("rssv.method.debug.drawTightAABB",true);
  auto&wireframe          = vars.addOrGetBool  ("rssv.method.debug.wireframe",true);

  auto&clearScreen        = vars.addOrGetBool("rssv.method.debug.clearScreen"     ,true);
  auto&drawSamples        = vars.addOrGetBool("rssv.method.debug.drawSamples"     );
  auto&drawNodePool       = vars.addOrGetBool("rssv.method.debug.drawNodePool"    );
  auto&drawTraverse       = vars.addOrGetBool("rssv.method.debug.drawTraverse"    );
  auto&drawShadowFrusta   = vars.addOrGetBool("rssv.method.debug.drawShadowFrusta");
  auto&drawEdges          = vars.addOrGetBool("rssv.method.debug.drawEdges"       );
  auto&drawSilhouettes    = vars.addOrGetBool("rssv.method.debug.drawSilhouettes" );
  auto&drawEdgePlanes     = vars.addOrGetBool("rssv.method.debug.drawEdgePlanes"  );
  auto&drawSVSides        = vars.addOrGetBool("rssv.method.debug.drawSVSides"     );


  auto&drawBridges        = vars.addOrGetBool  ("rssv.method.debug.drawBridges"  );
  auto&drawAllBridges     = vars.addOrGetBool  ("rssv.method.debug.drawAllBridges");
  auto&bridgesToDraw      = vars.addOrGetUint32("rssv.method.debug.bridgesToDraw",0x7);

  auto&drawStencil        = vars.addOrGetBool  ("rssv.method.debug.drawStencil"   ,true);

  auto&taToDraw = vars.addOrGetUint32("rssv.method.debug.taToDraw",0);
  auto&trToDraw = vars.addOrGetUint32("rssv.method.debug.trToDraw",0);
  auto&inToDraw = vars.addOrGetUint32("rssv.method.debug.inToDraw",0);

  if(ImGui::BeginMainMenuBar()){
    if(ImGui::BeginMenu("debug")){
      if(ImGui::MenuItem(clearScreen?"fillScreen":"clearScreen")){
        clearScreen = !clearScreen;
        vars.updateTicks("rssv.method.debug.clearScreen");
      }
      ImGui::EndMenu();
    }

    if(ImGui::BeginMenu("samples")){

      if(ImGui::MenuItem("dump")){
        rssv::debug::dumpBasic(vars);
        rssv::debug::dumpSamples(vars);
      }

      if(ImGui::MenuItem("draw")){
        drawSamples = !drawSamples;
        vars.updateTicks("rssv.method.debug.drawSamples");
      }

      ImGui::EndMenu();
    }

    if(ImGui::BeginMenu("nodes")){

      if(ImGui::MenuItem("dump")){
        rssv::debug::dumpBasic(vars);
        rssv::debug::dumpNodePool(vars);
        rssv::debug::dumpAABBPool(vars);
        rssv::debug::dumpAABBPointer(vars);
      }

      if(ImGui::MenuItem("drawNodePool")){
        drawNodePool = !drawNodePool;
        vars.updateTicks("rssv.method.debug.drawNodePool");
      }

      if(ImGui::MenuItem("drawTightAABB")){
        drawTightAABB = !drawTightAABB;
        vars.updateTicks("rssv.method.debug.drawTightAABB");
      }

      if(ImGui::MenuItem("wireframe")){
        wireframe = !wireframe;
        vars.updateTicks("rssv.method.debug.wireframe");
      }

      if(drawNodePool){
        if(vars.has("rssv.method.debug.dump.config")){
          auto const cfg = *vars.get<Config>        ("rssv.method.debug.dump.config"    );
          for(uint32_t i=0;i<cfg.nofLevels;++i){
            std::stringstream ss;
            ss << "level" << i;
            if(ImGui::MenuItem(ss.str().c_str())){
              levelsToDraw ^= 1<<i;
            }
          }
        }
      }

      ImGui::EndMenu();
    }

    if(ImGui::BeginMenu("bridges")){

      if(ImGui::MenuItem("dump","CTRL+D")){
        rssv::debug::dumpBasic(vars);
        rssv::debug::dumpNodePool(vars);
        rssv::debug::dumpAABBPool(vars);
        rssv::debug::dumpAABBPointer(vars);
        rssv::debug::dumpBridges(vars);
      }

      if(ImGui::MenuItem(drawBridges?"hide":"draw","CTRL+B")){
        drawBridges = !drawBridges;
        vars.updateTicks("rssv.method.debug.drawBridges");
      }
      if(ImGui::MenuItem(drawAllBridges?"hideEmpty":"all")){
        drawAllBridges = !drawAllBridges;
        vars.updateTicks("rssv.method.debug.drawAllBridges");
      }

      if(drawBridges){
        if(vars.has("rssv.method.debug.dump.config")){
          auto const cfg = *vars.get<Config>        ("rssv.method.debug.dump.config"    );
          for(uint32_t i=0;i<cfg.nofLevels;++i){
            std::stringstream ss;
            ss << (bridgesToDraw&(1<<i)?"hide":"show");
            ss << " level" << i;
            if(ImGui::MenuItem(ss.str().c_str())){
              bridgesToDraw ^= 1<<i;
            }
          }
        }
      }

      ImGui::EndMenu();
    }

    if(ImGui::BeginMenu("stencil")){

      if(ImGui::MenuItem(drawStencil?"hide":"draw")){
        drawStencil = !drawStencil;
        vars.updateTicks("rssv.method.debug.drawStencil");
      }

      ImGui::EndMenu();
    }


    if(ImGui::BeginMenu("silhouettes")){
      if(ImGui::MenuItem("dump")){
        rssv::debug::dumpBasic(vars);
        rssv::debug::dumpSilhouettes(vars);
      }

      if(ImGui::MenuItem("drawEdges")){
        drawEdges = !drawEdges;
        vars.updateTicks("rssv.method.debug.drawEdges");
      }


      if(ImGui::MenuItem("drawSilhouettes")){
        drawSilhouettes = !drawSilhouettes;
        vars.updateTicks("rssv.method.debug.drawSilhouettes");
      }

      if(ImGui::MenuItem("drawEdgePlanes")){
        drawEdgePlanes = !drawEdgePlanes;
        vars.updateTicks("rssv.method.debug.drawEdgePlanes");
      }
      if(ImGui::MenuItem("drawSVSides")){
        drawSVSides = !drawSVSides;
        vars.updateTicks("rssv.method.debug.drawSVSides");
      }

      ImGui::EndMenu();
    }


    if(ImGui::BeginMenu("traverse")){

      if(ImGui::MenuItem("dump traverse silhouettes data")){
        rssv::debug::dumpBasic(vars);
        rssv::debug::dumpNodePool(vars);
        rssv::debug::dumpAABBPool(vars);
        rssv::debug::dumpAABBPointer(vars);
        rssv::debug::dumpTraverseSilhouettes(vars);
      }

      if(ImGui::MenuItem("dump traverse triangles data")){
        rssv::debug::dumpBasic(vars);
        rssv::debug::dumpNodePool(vars);
        rssv::debug::dumpAABBPool(vars);
        rssv::debug::dumpAABBPointer(vars);
        rssv::debug::dumpTraverseTriangles(vars);
      }

      if(ImGui::MenuItem("dump planes")){
        rssv::debug::dumpBasic(vars);
        rssv::debug::dumpNodePool(vars);
        rssv::debug::dumpAABBPool(vars);
        rssv::debug::dumpTraversePlanes(vars);
      }


      if(ImGui::MenuItem(drawTraverse?"hide":"draw")){
        drawTraverse = !drawTraverse;
        vars.updateTicks("rssv.method.debug.drawTraverse");
      }

      if(ImGui::MenuItem("drawTightAABB")){
        drawTightAABB = !drawTightAABB;
        vars.updateTicks("rssv.method.debug.drawTightAABB");
      }

      if(drawTraverse){
        if(vars.has("rssv.method.debug.dump.config")){
          auto const cfg = *vars.get<Config>        ("rssv.method.debug.dump.config"    );
          for(uint32_t i=0;i<cfg.nofLevels;++i){
            std::stringstream ss;
            ss << (taToDraw&(1<<i)?"hide":"show");
            ss << " trivialAccept " << i;
            if(ImGui::MenuItem(ss.str().c_str())){
              taToDraw ^= 1<<i;
            }
          }
          for(uint32_t i=0;i<cfg.nofLevels;++i){
            std::stringstream ss;
            ss << (trToDraw&(1<<i)?"hide":"show");
            ss << " trivialReject " << i;
            if(ImGui::MenuItem(ss.str().c_str())){
              trToDraw ^= 1<<i;
            }
          }
          for(uint32_t i=0;i<cfg.nofLevels;++i){
            std::stringstream ss;
            ss << (inToDraw&(1<<i)?"hide":"show");
            ss << " intersect " << i;
            if(ImGui::MenuItem(ss.str().c_str())){
              inToDraw ^= 1<<i;
            }
          }
        }
      }

      ImGui::EndMenu();
    }


    if(ImGui::BeginMenu("dump")){

      if(ImGui::MenuItem("dump all"))
        rssv::debug::dumpData(vars);

      if(ImGui::MenuItem("drawShadowFrusta")){
        drawShadowFrusta = !drawShadowFrusta;
        vars.updateTicks("rssv.method.debug.drawShadowFrusta");
      }

      ImGui::EndMenu();
    }


    ImGui::EndMainMenuBar();
  }

  if(clearScreen){
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  }

  if(drawSamples)
    debug::drawSamples(vars);

  if(drawNodePool)
    debug::drawNodePool(vars);

  if(drawTraverse)
    debug::drawTraverse(vars);

  if(drawShadowFrusta)
    debug::drawSF(vars);

  if(drawEdges)
    debug::drawEdges(vars);

  if(drawSilhouettes)
    debug::drawSilhouettes(vars);

  if(drawEdgePlanes)
    debug::drawEdgePlanes(vars);

  if(drawSVSides)
    debug::drawSVSides(vars);

  if(drawBridges)
    debug::drawBridges(vars);

  if(drawStencil)
    debug::drawStencil(vars);

}
