#pragma once

#include <VSSV/DrawSides.h>
#include <geGL/geGL.h>
#include <Vars/Vars.h>

class Adjacency;
class DrawSidesUsingAllPlanes: public DrawSides{
  public:
    DrawSidesUsingAllPlanes(vars::Vars&vars,std::shared_ptr<Adjacency const>const&adj);
};
