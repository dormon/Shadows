#pragma once

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace ge 
{
    namespace gl 
    {
        class Program;
		class Shader;
    }
}


class FtsShaderGen
{
public:

    std::unique_ptr<ge::gl::Program> GetIzbFillProgram(glm::uvec3 const& bufferDims);
    std::unique_ptr<ge::gl::Program> GetIzbTraversalProgram(glm::uvec3 const& bufferDims);

    std::unique_ptr<ge::gl::Program> GetIzbFillProgramOmnidir(glm::uvec3 const& bufferDims);
    std::unique_ptr<ge::gl::Program> GetIzbTraversalProgramOmnidirRaytrace(glm::uvec3 const& bufferDims);
    std::unique_ptr<ge::gl::Program> GetIzbTraversalProgramOmnidirFrusta(glm::uvec3 const& bufferDims);

    std::unique_ptr<ge::gl::Program> GetTrianglePreprocessCS(uint32_t wgSize);

    std::unique_ptr<ge::gl::Program> GetEtsTraversalProgram(glm::uvec3 const& bufferDims);

	//Fucntions returning shaders
	std::vector<std::shared_ptr<ge::gl::Shader>> GetIzbFillProgramShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetIzbTraversalProgramShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetIzbFillProgramOmnidirShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetIzbTraversalProgramOmnidirRaytraceShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetIzbTraversalProgramOmnidirFrustaShaders(glm::uvec3 const& bufferDims);
	std::shared_ptr<ge::gl::Shader>	             GetTrianglePreprocessCSShader(uint32_t wgSize);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetEtsTraversalProgramShaders(glm::uvec3 const& bufferDims);

private:
 
    //Single-direction
    std::string GetFSFill(glm::uvec3 const& bufferDims) const;

    std::string GetVSTraversal() const;
    std::string GetFSTraversal(glm::uvec3 const& bufferDims) const;
    
    //Omnidirectional FTS
    std::string GetVSFillOmnidir() const;
    std::string GetGSFillOmnidir() const;
    std::string GetFSFillOmnidir(glm::uvec3 const& bufferDims) const;

    std::string GetFSTraversalOmnidirRaytrace(glm::uvec3 const& bufferDims) const;
    std::string GetFSTraversalOmnidirFrusta(glm::uvec3 const& bufferDims) const;

    std::string GetCSPreprocess(uint32_t wgSize) const;

    //Edge-traced
    std::string GetFsEtsFill(glm::uvec3 const& bufferDims) const;

    //Common
    std::string GetVSFill() const;

    std::string GetIzbAddressingFunction() const;
    std::string GetCubeAddressingFunction() const;
    std::string GetRayInersectFunction() const;
};