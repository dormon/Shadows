#include <imguiVars.h>

#include <imguiSDL2OpenGL/imgui.h>

#include<vector>
#include<map>
#include<iostream>

std::string getHead(std::string const&n){
  return n.substr(0,n.find("."));
}

std::string getTail(std::string const&n){
  return n.substr(n.find(".")+1);
}

bool hasSplit(std::string const&n){
  return n.find(".") != std::string::npos;
}

class Group{
  public:
    Group(std::string const&n,std::string const&fn = ""):name(n),fullName(fn){}
    std::string name;
    std::string fullName;
    std::map<std::string,std::unique_ptr<Group>>children;
};



class VarNamesHierarchy{
  public:
    std::map<std::string,std::unique_ptr<Group>>groups;
    VarNamesHierarchy(std::vector<std::string>const&names){
      for(auto const&name:names)
        insertName(name);
    }
    void insertName(std::string const&name){
      if(hasSplit(name)){
        auto const head = getHead(name);
        auto const tail = getTail(name);
        if(groups.count(head)==0)
          groups[head] = std::make_unique<Group>(head);
        insertToGroup(groups.at(head),tail,name);
      }else{
        groups[name] = std::make_unique<Group>(name,name);
      }

    }
    void insertToGroup(std::unique_ptr<Group>const&group,std::string const&name,std::string const&fullName){
      if(hasSplit(name)){
        auto const head = getHead(name);
        auto const tail = getTail(name);
        if(group->children.count(head)==0)
          group->children[head] = std::make_unique<Group>(head);
        insertToGroup(group->children.at(head),tail,fullName);
      }else{
        group->children[name] = std::make_unique<Group>(name,fullName);
      }
    }
};


void drawGroup(std::unique_ptr<Group>const&group,vars::Vars &vars){
  if(group->children.empty()){
    auto const n = group->name;
    auto const fn = group->fullName;
    bool change = false;
    if(vars.getType(fn) == typeid(float)){
      change = ImGui::DragFloat(n.c_str(),(float*)vars.get(fn));
    }
    if(vars.getType(fn) == typeid(uint32_t)){
      change = ImGui::InputScalar(n.c_str(),ImGuiDataType_U32,(uint32_t*)vars.get(fn));
      //ImGui::DragInt(n.c_str(),(int32_t*)vars.get(fn),1,0);
    }
    if(vars.getType(fn) == typeid(bool)){
      change = ImGui::Checkbox(n.c_str(),(bool*)vars.get(fn));
    }
    //if(vars.getType(fn) == typeid(std::string)){
    //  ImGui::TextV
    //}
    if(change)
      vars.updateTicks(fn);

  }else{
    if(ImGui::TreeNode(group->name.c_str())){
      for(auto const&x:group->children)
        drawGroup(x.second,vars);
      ImGui::TreePop();
    }
  }
}


void drawImguiVars(vars::Vars &vars){
  std::vector<std::string>names;
  for(size_t i = 0;i<vars.getNofVars();++i)
    names.push_back(vars.getVarName(i));
  
  VarNamesHierarchy hierarchy{names};
  

  ImGui::Begin("vars");

  for(auto const&x:hierarchy.groups)
    drawGroup(x.second,vars);

  ImGui::End();

  
}
