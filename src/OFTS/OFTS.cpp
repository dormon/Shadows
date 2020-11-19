#include "OFTS.h"

#include <BasicCamera/FreeLookCamera.h>
#include <ifExistStamp.h>
#include <FunctionPrologue.h>

#include <iostream>

OFTS::OFTS(vars::Vars& vars) : FTS(vars) {}

void OFTS::ComputeLightFrusta()
{
	FUNCTION_PROLOGUE("fts.objects", "lightPosition", "fts.args.nearZ", "fts.args.farZ");

	const float nearZ = vars.getFloat("fts.args.nearZ");
	const float farZ = vars.getFloat("fts.args.farZ");
	const float fovy = 1.5708f; //90 degrees
	const float aspectRatio = 1.f;
	glm::vec3 const lightPos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));

	glm::vec3 ups[] = {
		glm::vec3(0, 1, 0), //+x
		glm::vec3(0, 1, 0), //-x
		glm::vec3(0, 0, 1), //+y
		glm::vec3(0, 0,-1), //-y
		glm::vec3(0, 1, 0), //+z
		glm::vec3(0, 1, 0), //-z
	};

	glm::vec3 dirs[] = {
		glm::vec3(1, 0, 0), //+x
		glm::vec3(-1, 0, 0), //-x
		glm::vec3(0, 1, 0), //+y
		glm::vec3(0,-1, 0), //-y
		glm::vec3( 0, 0, 1), //+z
		glm::vec3( 0, 0,-1), //-z
	};

	for(uint32_t i = 0; i<6; ++i)
	{
		lightFrusta[i] = Frustum(fovy, aspectRatio, nearZ, farZ, lightPos, lightPos + dirs[i], ups[i]);
	}
}

Frustum OFTS::GetCameraFrustum(glm::mat4 const& viewMatrix) const
{
	float const fovy = vars.getFloat("args.camera.fovy");
	float const near = vars.getFloat("args.camera.near");
	float far = vars.getFloat("args.camera.far");
	far = glm::isinf(far) != 0 ? 50000.f : far;
	
	glm::vec2 const windowSize = glm::vec2(*vars.get<glm::uvec2>("windowSize"));
	const float aspectRatio = windowSize.x / windowSize.y;

	glm::vec3 const pos = vars.get<basicCamera::FreeLookCamera>("cameraTransform")->getPosition();
	glm::vec3 const up = glm::vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
	glm::vec3 const forward = -glm::vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);
	
	return Frustum(fovy, aspectRatio, near, far, pos, pos + forward, up);
}

uint8_t OFTS::GetUsedFrustumMasks(glm::mat4 const& viewMatrix) const
{
	uint8_t mask = 0;

	Frustum const camFrustum = GetCameraFrustum(viewMatrix);

	for(uint8_t i =0; i<6; ++i)
	{
		if(camFrustum.isFrustumIntersecting(lightFrusta[i]))
		{
			mask |= (uint8_t(1) << i);
		}
	}

	return mask;
}

glm::mat4 OFTS::GetLightViewMatrix(uint8_t index) const
{
	assert(index < 6);

	glm::vec3 ups[] = {
		glm::vec3(0, 1, 0), //+x
		glm::vec3(0, 1, 0), //-x
		glm::vec3(0, 0, 1), //+y
		glm::vec3(0, 0,-1), //-y
		glm::vec3(0, 1, 0), //+z
		glm::vec3(0, 1, 0), //-z
	};

	glm::vec3 dirs[] = {
		glm::vec3(1, 0, 0), //+x
		glm::vec3(-1, 0, 0), //-x
		glm::vec3(0, 1, 0), //+y
		glm::vec3(0,-1, 0), //-y
		glm::vec3(0, 0, 1), //+z
		glm::vec3(0, 0,-1), //-z
	};

	glm::vec3 const lightPos = glm::vec3(*vars.get<glm::vec4>("lightPosition"));

	return glm::lookAt(lightPos, lightPos + dirs[index], ups[index]);
}

glm::mat4 OFTS::GetLightProjMatrix() const
{
	const float nearZ = vars.getFloat("fts.args.nearZ");
	const float farZ = vars.getFloat("fts.args.farZ");
	const float fovy = 1.5708f; //90 degrees
	const float aspectRatio = 1.f;

	return glm::perspective(fovy, aspectRatio, nearZ, farZ);
}

void OFTS::PrintStats(uint8_t mask) const
{
	static uint8_t prevMask = 0;

	if (prevMask != mask)
	{
		prevMask = mask;
		int nofBits = 0;
		for (int b = 0; b < 6; ++b)
		{
			if ((mask >> b) & 1)
			{
				++nofBits;
			}
		}
		std::cout << uint32_t(mask) << " (" << nofBits << "): ";

		for (uint32_t i = 0; i < 6; ++i)
		{
			if ((mask >> i) & 1)
			{
				switch (i)
				{
				case 0: std::cout << "+x "; break;
				case 1: std::cout << "-x "; break;
				case 2: std::cout << "+y "; break;
				case 3: std::cout << "-y "; break;
				case 4: std::cout << "+z "; break;
				case 5: std::cout << "-z "; break;
				default:
					break;
				}
			}
		}

		std::cout << std::endl;
	}
}

void OFTS::create(glm::vec4 const& lightPosition, glm::mat4 const& viewMatrix, glm::mat4 const& projectionMatrix)
{
	if (!IsValid())
	{
		return;
	}
	
	CreateDummyVao();
	CreateShadowMaskVao();
	CreateShadowMaskFbo();
	CompileShaders(true);
	CreateTextures();
	CreateBuffers();
	ComputeLightFrusta();

	ClearShadowMask();

	uint8_t const mask = GetUsedFrustumMasks(viewMatrix);
	//PrintStats(mask);

	glm::mat4 const lightP = GetLightProjMatrix();
	glm::mat4 const vp = projectionMatrix * viewMatrix;

	ifExistStamp("");

	for(uint8_t i = 0; i< 6; ++i)
	{
		if(((mask >> i) & 1) == 0)
		{
			continue;
		}

		glm::mat4 const lightV = GetLightViewMatrix(i);
		glm::mat4 const lightVP = lightP * lightV;

		ClearTextures();

		ComputeHeatMap(lightVP);

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		ComputeViewProjectionMatrix();

		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		ComputeIzb(vp, lightV);

		//ifExistStamp("ftsCreate");

		InitShadowMaskZBuffer();

		//ifExistStamp("ftsZFill");

		FillShadowMask(lightV);
	}

	ifExistStamp("ofts");
}
