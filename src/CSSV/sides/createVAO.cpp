#include <geGL/geGL.h>

#include <FunctionPrologue.h>
#include <ShadowMethod.h>
#include <FastAdjacency.h>
#include <divRoundUp.h>

#include <CSSV/sides/createVAO.h>

using namespace ge::gl;

void cssv::sides::createVAO(vars::Vars&vars){
  FUNCTION_PROLOGUE("cssv.method"
      ,"cssv.method.silhouettes"
      ,"cssv.param.dontExtractMultiplicity"
      );

  bool const dontExtMult = vars.getBool("cssv.param.dontExtractMultiplicity");

  if(dontExtMult){
    auto silhouettes = vars.get<Buffer>("cssv.method.silhouettes");
    auto vao = vars.reCreate<VertexArray>("cssv.method.sides.vao");
    vao->addAttrib(silhouettes,0,componentsPerVertex4D,GL_FLOAT);
  }else{
    vars.reCreate<VertexArray>("cssv.method.sides.vao");
    std::vector<float>edges;
    auto adj = vars.get<Adjacency>("adjacency");
    std::cerr << "nofEdges: " << adj->getNofEdges() << std::endl;
    auto&vert = adj->getVertices();
    //for(size_t e=0;e<adj->getNofEdges();++e){
    //  edges.push_back(vert[adj->getEdgeVertexA(e)+0]);
    //  edges.push_back(vert[adj->getEdgeVertexA(e)+1]);
    //  edges.push_back(vert[adj->getEdgeVertexA(e)+2]);
    //  edges.push_back(vert[adj->getEdgeVertexB(e)+0]);
    //  edges.push_back(vert[adj->getEdgeVertexB(e)+1]);
    //  edges.push_back(vert[adj->getEdgeVertexB(e)+2]);
    //}

    auto nofE = adj->getNofEdges();
    auto anofE = divRoundUp(nofE,1024)*1024;
    edges.resize(anofE*6,0);
    for(size_t e=0;e<adj->getNofEdges();++e)edges[e+0*anofE] = vert[adj->getEdgeVertexA(e)+0];
    for(size_t e=0;e<adj->getNofEdges();++e)edges[e+1*anofE] = vert[adj->getEdgeVertexA(e)+1];
    for(size_t e=0;e<adj->getNofEdges();++e)edges[e+2*anofE] = vert[adj->getEdgeVertexA(e)+2];
    for(size_t e=0;e<adj->getNofEdges();++e)edges[e+3*anofE] = vert[adj->getEdgeVertexB(e)+0];
    for(size_t e=0;e<adj->getNofEdges();++e)edges[e+4*anofE] = vert[adj->getEdgeVertexB(e)+1];
    for(size_t e=0;e<adj->getNofEdges();++e)edges[e+5*anofE] = vert[adj->getEdgeVertexB(e)+2];
    vars.reCreate<uint32_t>("cssv.method.alignedNofEdges",anofE);
    
    vars.reCreate<Buffer>("cssv.method.edgeBuffer",edges);
  }
}

