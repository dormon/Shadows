#include <RayTracing/RayTracing.h>
#include <RayTracing/nanort.h>

#include<algorithm>
#include<numeric>

#include<Deferred.h>
#include<Model.h>

using namespace ge::gl;
using namespace std;

#include<Vars/Resource.h>

#include<Barrier.h>

#define ___ std::cerr << __FILE__ << " " << __LINE__ << std::endl

void buildBVH(vars::Vars&vars){
  if(notChanged(vars,"rayTracing",__FUNCTION__,{"model"}))return;
  ___;

  vector<float>vertices = vars.get<Model>("model")->getVertices();
  ___;
 

  nanort::BVHBuildOptions<float> build_options;  // Use default option
  ___;
  build_options.cache_bbox = false;
  ___;

  std::vector<uint32_t>faces(vertices.size());
  ___;
  std::iota(faces.begin(),faces.end(),0);
  ___;

  nanort::TriangleMesh<float> triangle_mesh(vertices.data(), faces.data(),
                                            sizeof(float) * 3);
  ___;
  nanort::TriangleSAHPred<float> triangle_pred(vertices.data(), faces.data(),
                                               sizeof(float) * 3);
  ___;

  typedef nanort::BVHAccel<float> Accel;
  ___;
  auto accel = vars.reCreate<Accel>("accel");
  ___;
  auto ret = accel->Build((uint32_t)vertices.size()/9, triangle_mesh, triangle_pred, build_options);
  ___;
  assert(ret);
  ___;


  nanort::BVHBuildStatistics stats = accel->GetStatistics();
  ___;

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  accel->BoundingBox(bmin, bmax);
  ___;
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);



  auto nodes = accel->GetNodes();
  for(auto const&n:nodes){
    if(n.flag)std::cerr << "leaf" << std::endl;
    else std::cerr << "branch" << std::endl;
    std::cerr << n.bmin[0] << " " << n.bmin[1] << " " << n.bmin[2] << std::endl;
    std::cerr << n.bmax[0] << " " << n.bmax[1] << " " << n.bmax[2] << std::endl;
    if(n.flag){
      std::cerr << "npoints: " << n.data[0] << " index: " << n.data[1] << std::endl;
    }else{
      std::cerr << "child[0]: " << n.data[0] << " child[1]: " << n.data[1] << std::endl;
    }
    std::cerr << "axis: " << n.axis << std::endl;
  }

}

RayTracing::RayTracing(vars::Vars&vars):ShadowMethod(vars){
  buildBVH(vars);


}

RayTracing::~RayTracing(){
  vars.erase("rayTracing");
}

void RayTracing::create(
    glm::vec4 const&lightPosition,
    glm::mat4 const&             ,
    glm::mat4 const&             ){
  ifExistStamp("");
  ifExistStamp("raytrace");
}

