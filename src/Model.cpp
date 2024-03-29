#include<Model.h>

#include<GLSLLine.h>

Model::Model(std::string const&fileName)
{
	model = aiImportFile(fileName.c_str(),aiProcess_Triangulate|aiProcess_GenNormals|aiProcess_SortByPType);

	if (model == nullptr)
	{
		std::cerr << "Can't open file: " << fileName << std::endl;
	}
	else
	{
		generateVertices();
		name = model->GetShortFilename(fileName.c_str());
	}
}

Model::~Model(){
  assert(this!=nullptr);
  if(this->model)aiReleaseImport(this->model);
}

std::vector<float> Model::getVertices() const
{
	return vertices;
}

void Model::generateVertices(){
  size_t nofVertices = 0;
  for(size_t i=0;i<model->mNumMeshes;++i)
    nofVertices+=model->mMeshes[i]->mNumFaces*3;
  vertices.reserve(nofVertices*3);
  for(size_t i=0;i<model->mNumMeshes;++i){
    auto mesh = model->mMeshes[i];
    for(size_t j=0;j<mesh->mNumFaces;++j)
      for(size_t k=0;k<3;++k)
        for(size_t l=0;l<3;++l)
          vertices.push_back(mesh->mVertices[mesh->mFaces[(uint32_t)j].mIndices[(uint32_t)k]][(uint32_t)l]);
  }
}

RenderModel::RenderModel(Model*mdl){
  assert(this!=nullptr);
  if(mdl==nullptr)
    std::cerr << "mdl is nullptr!" << std::endl;

  this->nofVertices = 0;
  auto model = mdl->model;
  for(size_t i=0;i<model->mNumMeshes;++i)
    this->nofVertices+=model->mMeshes[i]->mNumFaces*3;

  std::vector<float>vertData;
  vertData = mdl->getVertices();
  this->vertices = std::make_shared<ge::gl::Buffer>(this->nofVertices*sizeof(float)*3,vertData.data());

  std::vector<float>normData;
  normData.reserve(this->nofVertices*3);
  for(size_t i=0;i<model->mNumMeshes;++i){
    auto mesh = model->mMeshes[i];
    for(uint32_t j=0;j<mesh->mNumFaces;++j)
      for(uint32_t k=0;k<3;++k)
        for(uint32_t l=0;l<3;++l)
          normData.push_back(mesh->mNormals[mesh->mFaces[j].mIndices[k]][l]);
  }
  this->normals = std::make_shared<ge::gl::Buffer>(this->nofVertices*sizeof(float)*3,normData.data());


/*
  {
    std::vector<float>ver;
    for(size_t i=0;i<model->mNumMeshes;++i){
      auto mesh = model->mMeshes[i];
      for(size_t j=0;j<mesh->mNumVertices;++j){
        for(size_t k=0;k<3;++k)
          ver.push_back(mesh->mVertices[j][k]);
        for(size_t k=0;k<3;++k)
          ver.push_back(mesh->mNormals[j][k]);
      }
    }
    std::vector<uint32_t>ind;
    uint32_t offset=0;
    for(size_t i=0;i<model->mNumMeshes;++i){
      auto mesh = model->mMeshes[i];
      for(size_t j=0;j<mesh->mNumFaces;++j)
        for(size_t k=0;k<3;++k)
          ind.push_back(offset+mesh->mFaces[j].mIndices[k]);
      offset+=mesh->mNumFaces*3;
    }
    this->indexVertices = std::make_shared<ge::gl::Buffer>(ver.size()*sizeof(float),ver.data());
    this->indices = std::make_shared<ge::gl::Buffer>(ind.size()*sizeof(uint32_t),ind.data());
    this->vao = std::make_shared<ge::gl::VertexArray>();
    this->vao->addAttrib(this->indexVertices,0,3,GL_FLOAT,sizeof(float)*6,0);
    this->vao->addAttrib(this->indexVertices,1,3,GL_FLOAT,sizeof(float)*6,sizeof(float)*3);
    this->vao->addElementBuffer(this->indices);
    this->nofVertices = ind.size();
  }
*/
  //*
  this->vao = std::make_shared<ge::gl::VertexArray>();
  this->vao->addAttrib(this->vertices,0,3,GL_FLOAT);
  this->vao->addAttrib(this->normals,1,3,GL_FLOAT);
  // */

  const std::string vertSrc =
"#version 450 \n"
GLSL_LINE
R".(
    uniform mat4 projectionView = mat4(1);

  layout(location=0)in vec3 position;
  layout(location=1)in vec3 normal;

  out vec3 vPosition;
  out vec3 vNormal;

  flat out uint vID;

  void main(){
    vID = gl_VertexID/3;
    gl_Position = projectionView*vec4(position,1);
    vPosition = position;
    vNormal   = normal;
  }).";
  const std::string fragSrc = 
"#version 450\n" 
GLSL_LINE
R".(
  layout(location=0)out uvec4 fColor;
  layout(location=1)out vec4  fPosition;
  layout(location=2)out vec4  fNormal; 
  layout(location=3)out uint  fTriangleID;

  in vec3 vPosition;
  in vec3 vNormal;

  flat in uint vID;
  vec3 hue(float t){
    t = fract(t);
    if(t<1/6.)return mix(vec3(1,0,0),vec3(1,1,0),(t-0/6.)*6);
    if(t<2/6.)return mix(vec3(1,1,0),vec3(0,1,0),(t-1/6.)*6);
    if(t<3/6.)return mix(vec3(0,1,0),vec3(0,1,1),(t-2/6.)*6);
    if(t<4/6.)return mix(vec3(0,1,1),vec3(0,0,1),(t-3/6.)*6);
    if(t<5/6.)return mix(vec3(0,0,1),vec3(1,0,1),(t-4/6.)*6);
              return mix(vec3(1,0,1),vec3(1,0,0),(t-5/6.)*6);
  }

  void main(){

    //fTriangleID = vID;
    vec3 colors[6] = vec3[6](
        vec3(1, 0, 0), 
        vec3(0, 1, 0),
        vec3(0, 0, 1),
        vec3(1, 1, 0),
        vec3(0, 1, 1),
        vec3(1, 0, 1)
    );


    vec3  diffuseColor   = colors[(vID /3) % 6];
    vec3  specularColor  = vec3(1);
    vec3  normal         = normalize(gl_FrontFacing ? vNormal : -vNormal);
    float specularFactor = 1;

    uvec4 color  = uvec4(0);
    color.xyz   += uvec3(diffuseColor  *0xff);
    color.xyz   += uvec3(specularColor *0xff)<<8;
    color.w      = uint (specularFactor*0xff);

    fColor    = color;	
    fPosition = vec4(vPosition,1);
    fNormal   = vec4(normal,-dot(vPosition,normal));
  }).";
  auto vs = std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, vertSrc);
  auto fs = std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fragSrc);
  this->program = std::make_shared<ge::gl::Program>(vs,fs);
}

RenderModel::~RenderModel(){
  assert(this!=nullptr);
}


void RenderModel::draw(glm::mat4 const&projectionView){
  assert(this!=nullptr);
  this->vao->bind();
  this->program->use();
  this->program->setMatrix4fv("projectionView",glm::value_ptr(projectionView));
  //this->glDrawElements(GL_TRIANGLES,this->nofVertices,GL_UNSIGNED_INT,nullptr);
  this->glDrawArrays(GL_TRIANGLES,0,this->nofVertices);
  this->vao->unbind();
}
