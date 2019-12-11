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


class DpmShaderGen
{
public:

    std::unique_ptr<ge::gl::Program> GetDpmFillProgram(glm::uvec3 const& bufferDims);
    std::unique_ptr<ge::gl::Program> GetDpmTraversalProgram(glm::uvec3 const& bufferDims);

    std::unique_ptr<ge::gl::Program> GetDpmFillProgramOmnidir(glm::uvec3 const& bufferDims);
    std::unique_ptr<ge::gl::Program> GetDpmTraversalProgramOmnidirRaytrace(glm::uvec3 const& bufferDims);
    std::unique_ptr<ge::gl::Program> GetDpmTraversalProgramOmnidirFrusta(glm::uvec3 const& bufferDims);

    std::unique_ptr<ge::gl::Program> GetTrianglePreprocessCS(uint32_t wgSize);

    std::unique_ptr<ge::gl::Program> GetEtsTraversalProgram(glm::uvec3 const& bufferDims);

	//Fucntions returning shaders
	std::vector<std::shared_ptr<ge::gl::Shader>> GetDpmFillProgramShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetDpmTraversalProgramShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetDpmFillProgramOmnidirShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetDpmTraversalProgramOmnidirRaytraceShaders(glm::uvec3 const& bufferDims);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetDpmTraversalProgramOmnidirFrustaShaders(glm::uvec3 const& bufferDims);
	std::shared_ptr<ge::gl::Shader>	             GetTrianglePreprocessCSShader(uint32_t wgSize);
	std::vector<std::shared_ptr<ge::gl::Shader>> GetEtsTraversalProgramShaders(glm::uvec3 const& bufferDims);

private:
 
    //Single-direction
    std::string GetFSFill(glm::uvec3 const& bufferDims) const;

    std::string GetVSTraversal() const;
    std::string GetFSTraversal(glm::uvec3 const& bufferDims) const;
    
    //Omnidirectional DPM
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

    std::string GetDpmAddressingFunction() const;
    std::string GetCubeAddressingFunction() const;
    std::string GetRayInersectFunction() const;
};