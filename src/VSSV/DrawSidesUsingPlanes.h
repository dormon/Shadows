#pragma once

#include <VSSV/DrawSides.h>
#include <geGL/geGL.h>
#include <Vars/Fwd.h>

class Adjacency;
class DrawSidesUsingPlanes: public DrawSides{
  public:
    DrawSidesUsingPlanes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};
