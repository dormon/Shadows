#include <DpsvShaders.h>
#include <sstream>
#include <string>

//All shaders based on https://github.com/PSVcode/EGSR2016

#define USE_OPTIMIZATION

std::string const vsSource = R".(
#version 450 core

void main()
{
	gl_Position = vec4(-1 + 2*(gl_VertexID / 2), -1 + 2*(gl_VertexID % 2), 0, 1);
}
).";

std::string const fsPrologue = R".(
#version 430

layout( binding = 0) uniform sampler2D positionTex;

layout(location=0) out float fColor;

struct Node 
{
	vec4 plane;
	uint link[4]; /* 0: positive child, 1: intersection child, 2: negative child (not used), 3: wedge angle */
};

// TOP tree buffer.
layout (std430, binding=0) buffer TOPTree	{ Node nodes[]; };

// Buffer to read the root index
layout (std430, binding=1) buffer TOPTreeRoot	{ uint root; };

uniform vec4 lightPosition;
).";

std::string const fsEpilogue = R".(
// must return the fragment position in the world space coordinates system
vec4 getFragmentPosition()
{
	const ivec2 tcoord = ivec2(gl_FragCoord.xy);
	return texelFetch(positionTex,tcoord,0);
}

// return true if a fragment exists, otherwise false (no projected geometry)
bool fragmentExists( in vec4 frag )
{
	return frag.w != 0;
	//return true;
}

void main()
{	
	vec4 pos = getFragmentPosition();
	float visibility = 0.f;
	if(fragmentExists(pos))
	{
		visibility = traverseToptree(pos.xyz-lightPosition.xyz); 
	}
	
	fColor = visibility;
}
).";

std::shared_ptr<ge::gl::Shader> getDpsvBuildCS(unsigned int wgSize, bool enableFrontFaceCulling, bool enableDepthOptim)
{
	std::stringstream str;
	str << "#version 430 core\n";
	str << "layout (local_size_x = " << wgSize << ",local_size_y = 1,local_size_z = 1) 	in;\n";

	if(enableFrontFaceCulling)
	{
		str << "#define FFCULLING\n";
	}

	str << R".(

struct Triangle {
	vec3 a;
	vec3 b;
	vec3 c;
};

struct Node
{
	vec4 plane;
	uint link[4]; /* 0: positive child, 1: intersection child, 2: negative child (not used), 3: wedge angle */
};

layout (std430, binding=0) buffer _vertices{ float vertices[];};
layout (std430, binding=1) buffer TOPTree	{ Node nodes[]; };

layout (std430, binding=2)  buffer util
{
	//uint globroot; 
	//uint globnode; // initialized to 4
	//uint globtriangle;  // initialized to 0
	//uint globpadding;
	uint globData[];
};

uniform vec4 lightPosition;
uniform uint nofTriangles;
uniform float bias = 0.0001f;

#define NOF_FLOATS_TRIANGLE 9u

// must return the i th triangle in the world space coordinates system
Triangle getTriangle( in uint i )
{
	const uint startingIndex = NOF_FLOATS_TRIANGLE * i;
	Triangle t;

	t.a = vec3(vertices[startingIndex + 0], vertices[startingIndex + 1], vertices[startingIndex + 2]);
	t.b = vec3(vertices[startingIndex + 3], vertices[startingIndex + 4], vertices[startingIndex + 5]);
	t.c = vec3(vertices[startingIndex + 6], vertices[startingIndex + 7], vertices[startingIndex + 8]);

	return t;
}

// return the plane defined by the light and the segment v1v2 (assuming the light is the origin)
vec4 computeShadowPlane( in vec3 v1, in vec3 v2)
{
	if ( v1.x < v2.x ) // partial test, but generally it is sufficient in practice. Otherwise y-axis and z-axis has to be tested
	{
		return vec4(  normalize( cross(v1, v2-v1) ), 0.0f);
	}
	
	return vec4( -normalize( cross(v2, v1-v2) ), 0.0f);
}

int trianglePosition(in vec3 A, in vec3 B, in vec3 C, in vec4 plane)
{
	const int sig = int(sign( dot(plane, vec4(A, 1)) )) +
					int(sign( dot(plane, vec4(B, 1)) )) +
					int(sign( dot(plane, vec4(C, 1)) )) ;

	return abs(sig)==3 ? sig : 0;
}

float wedgeAngle( in vec4 plane, in vec3 A, in vec3 B, in vec3 C)
{
	float d1 = dot(plane, vec4(A, 1)); // distance from A to the shadow plane
	float d2 = dot(plane, vec4(B, 1)); // distance from B to the shadow plane
	float d3 = dot(plane, vec4(C, 1)); // distance from C to the shadow plane
	// recall that a shadow plane contains the light
	d1 = d1*d1 / dot(A,A); // squared sine of the angle between the shadow plane and the segment lightA
	d2 = d2*d2 / dot(B,B); // squared sine of the angle between the shadow plane and the segment lightB
	d3 = d3*d3 / dot(C,C); // squared sine of the angle between the shadow plane and the segment lightC
	
    return(max(d1,max(d2,d3))); // return the maximum of the 3 (squared) sines
}

float dot2( in vec3 v ) 
{ 
	return dot(v,v); 
}

float tri2LightDistance( in vec3 v1, in vec3 v2, in vec3 v3, in vec3 p )
{
    const vec3 v21 = v2 - v1; vec3 p1 = p - v1;
    const vec3 v32 = v3 - v2; vec3 p2 = p - v2;
    const vec3 v13 = v1 - v3; vec3 p3 = p - v3;
    const vec3 nor = cross( v21, v13 );

    return sqrt( (sign(dot(cross(v21,nor),p1)) + 
                  sign(dot(cross(v32,nor),p2)) + 
                  sign(dot(cross(v13,nor),p3))<2.0) 
                  ?
                  min( min( 
                  dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0,1.0)-p1), 
                  dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0,1.0)-p2) ), 
                  dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0,1.0)-p3) )
                  :
                  dot(nor,p1)*dot(nor,p1)/dot2(nor) );
}

void TOPTREE_mergeShadowVolumeCastByTriangle( in uint i )
{
		const vec3 light = lightPosition.xyz;
		
		const Triangle T = getTriangle(i);
		// vertices translation to make the light the origin (this is only to simplify the computations)
		const vec3 A  = T.a.xyz - light;
		const vec3 B  = T.b.xyz - light;
		const vec3 C  = T.c.xyz - light;
		// compute the capping plane equation (i.e. the supporting plane of triangle ABC)
		const vec3 norm = normalize( cross(C-A, B-A) );
		const vec4 capping_plane = vec4( norm, -dot(norm,A) );

#ifdef FFCULLING
		if ( capping_plane.w < 0.0 ) // Front Face Culling enable
#endif
		{
			// book 4 nodes in the TOP tree buffer to represent the SV generated by the light and triangle ABC 
			const uint insertion = atomicAdd(globData[1], 4);
			Node sp1,sp2,sp3,cp;
			// initialize the 4 nodes with the 3 shadow planes and the capping plane
			sp1.plane 		= computeShadowPlane(A, B);
			sp2.plane 		= computeShadowPlane(B, C);
			sp3.plane 		= computeShadowPlane(C, A);
			cp.plane 	    = -capping_plane;

#ifndef FFCULLING 
			if ( capping_plane.w > 0.0 ) // correct SV orientation for triangles front facing the light
			{
				sp1.plane = -sp1.plane;
				sp2.plane = -sp2.plane;
				sp3.plane = -sp3.plane;
				cp.plane  = -cp.plane;
			}	
#endif
			// slightly translate the capping plane away from the light to get ride of self shading artifacts
			cp.plane.w += bias;
).";

	if (enableDepthOptim)
	{
		str << "			const uint distance = floatBitsToUint( tri2LightDistance(A, B, C, light) );\n";
	}
	else
	{
		str << "			const uint distance = 0;\n";
	}
	
	str << R".(
			// init nodes
			sp1.link[0] = 0;		sp1.link[1] = 0;		sp1.link[2] = distance; 		sp1.link[3] = 0;
			sp2.link[0] = 0;		sp2.link[1] = 0;		sp2.link[2] = 0;  	       		sp2.link[3] = 0;
			sp3.link[0] = 0;		sp3.link[1] = 0;		sp3.link[2] = 0;        	    sp3.link[3] = 0;
			cp.link[0]  = 0;	    cp.link[1]  = 0;    	cp.link[2]  = 0;			    cp.link[3]  = 0;
			
			// write the nodes in the array. Notice that this four node are connected by their negative child. 
			// However we do not use link[2]. Instead we will compute the negative index on the fly to avoid a buffer read
			nodes[insertion  ] = sp1;
			nodes[insertion+1] = sp2;
			nodes[insertion+2] = sp3;
			nodes[insertion+3] = cp;

			// if root equals 0, the TOP tree is empty and root is replaced by insertion that becomes the new root glob
			// Otherwise we simply get the root index of the TOP tree
			uint current = atomicCompSwap(globData[0], 0, insertion);

			// [EGSR2016] - stackless support - holds the parent node of the last intersection subtree visited by the triangle
			uint lastsubroot = 1;

			// find the triangle location (except if the tree was empty)
			while( current != 0)
			{
				// compute the triangle position wrt the current plane
				const int pos = trianglePosition(A, B, C, nodes[current].plane);
).";
	if (enableDepthOptim)
	{
		str << R".(
				// [EGSR2106] - depth test support - update the distance from the light each time a new shadow volume is visited
				if (current%4==0)
					atomicMin(nodes[current].link[2], distance);
).";
	}

	str << R".(		
				if(pos<0) // the triangle is fully in the negative halfspace, compute the negative index
					if (current%4==3) current=0; // if the negative child is a leaf, the triangle is inside a shadow volume. This is an early termination case without merging the shadow volume.
					else ++current;	// otherwise, continue in the negative child	
				else
					if(pos>0) // the triangle is fully in the positive halfspace
						// if link[0] equals 0, the positive child is a leaf. Thus 0 is replaced by insertion, merging the shadow volume
						// cast by ABC. Otherwise we simply get the positive child index.
						current = atomicCompSwap(nodes[current].link[0], 0, insertion);
					else // the current plane intersects the triangle 
					{
						// [EGSR2016] - stackless support - the triangle descends in an intersection subtree, update lastsubroot
						lastsubroot = current;
						if ( current%4<3 ) // if the current plane is a shadow plane (wedge optimization is not relevant for the capping plane)
							atomicMax(nodes[current].link[3], floatBitsToUint(wedgeAngle(nodes[current].plane, A, B, C)));	// update the wedge angle
						// continue in the intersection child. If it equals 0, it is a leaf. Thus 0 is replaced by insertion, merging the shadow volume
						// cast by ABC. Otherwise, we simply get the intersection child index
						current = atomicCompSwap(nodes[current].link[1], 0, insertion);						
					}
			}
			// [EGSR2016] - stackless support - write the parent node index of the last intersection subtree visited by the triangle ABC
			nodes[insertion+2].link[2] = lastsubroot;
		}
}

/*
  Very rough persistant style variation. A thread merges triangles as long as triangles remain.
  We use a glDispatchCompute(32,1,1) with this one. 
*/
void main(void)	
{
	for(uint k=0;k<nofTriangles;k++)
	{
		uint i = atomicAdd(globData[2], 1);

		if (i>=nofTriangles) 
		{
			break;
		}

		TOPTREE_mergeShadowVolumeCastByTriangle(i);
	}
}
).";
	auto s = str.str();
	return std::make_shared<ge::gl::Shader>(GL_COMPUTE_SHADER, str.str());
}

std::shared_ptr<ge::gl::Shader> getDpsvVertexShader()
{
	return std::make_shared<ge::gl::Shader>(GL_VERTEX_SHADER, vsSource);
}

std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvStackProgramShaders(bool enableDepthOptim)
{
	std::stringstream str;
	str << R".(
float traverseToptree( in vec3 p)
{
	uint stacksize = 0;
	// node stack
    uint stack[32];
    // light - p distance
	const float dist = length(p);

 	// start from root node index
	uint current = root;

	// find the location of p
	while(current>1)
	{
		// pop
		const Node n = nodes[ current ];
).";
	if(enableDepthOptim)
	{
		str << R".(
		// [EGSR2016] skip the current subtree if p is closest from the light than the geometry in the subtree
		if (current % 4 == 0 && dist < uintBitsToFloat(n.link[2]))
		{
			current = stacksize > 0 ? stack[--stacksize] : 1;
			continue;
		}
).";
	}
	str << R".(
		// compute the signed distance from p to the current plane
		const float offset = dot(n.plane.xyz, p) + n.plane.w;

		// if an intersection child exists and if current is a capping plane or if it is a shadow plane and p is inside its wedge 
		if ( n.link[1]>0  &&  (current%4==3 || offset*offset / (dist*dist) < uintBitsToFloat(n.link[3] )  ) )
		   stack[ stacksize++ ] = n.link[1];
				
		// if p is in the positive halfspace of the plane
		if ( offset>0.0 ){
			// continue in the positive child if it exists	
			if (n.link[0]>0) current = n.link[0] ;
			// pop another node from the stack. If empty end the query with 1 (outside all shadow volumes)
			else current = stacksize>0 ? stack[ --stacksize ] : 1;										
		}
		else // otherwise p is in the negative halfspace
			// 0, p is inside a shadow volume. Otherwise continue in the negative child
			current = current%4==3 ? 0 : current+1;	
			
	}

	return float(current);
}

).";

	return { getDpsvVertexShader(), 
		std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fsPrologue + str.str() + fsEpilogue) };
}

std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvStacklessProgramShaders(bool enableDepthOptim)
{
	std::stringstream str;
	str << R".(
float traverseToptree( in vec3 p)
{ //avec distance uint
	// true : first time we visit the current node. False : second time we visit the current node
	bool secondVisit = false;
	// distance from p to the light
	const float dist =  length(p);

 	// start from root node index
	uint current = root;

	// find the location of p
	while(current>1){
		
		const Node n = nodes[ current ];
		
).";

	if (enableDepthOptim)
	{
		str << R".(
		// skip the current subtree if p is closest from the light than the geometry in the subtree
		if ( current%4==0 && dist< uintBitsToFloat(n.link[2]) ) {
			current = nodes[current+2].link[2];
			secondVisit = true;
		}
		else
).";
	}
	str << R".(	
		{
			// signed distance from p to the current plane
			const float offset = dot(n.plane.xyz, p) + n.plane.w;
	
			// if this is our first visit, if an intersection child exists and if current is a capping plane or if it is a shadow plane and p is inside its wedge
	 		if ( secondVisit==false && n.link[1]>0u && offset*offset / (dist*dist)< uintBitsToFloat(n.link[3])  )
	 			current = n.link[1]; // continue in the intersection child
	 		else // continue either in the positive child or negative child
	 		{
				if ( offset>0.0f ){ // go left
					if (n.link[0]==0){
						// we reach a positive leaf without finding any occlusion in this subtree, we are about to jump back in the tree
						// the next visited node has already been visited
						secondVisit=true;
						// jump back to the parent node of the last intersection child we met
						current = nodes[current - current%4 + 2].link[2];
					}
					else{ // positive child
						secondVisit=false;
						current = n.link[0];
					}
				
				}
				else { // negative child
					secondVisit=false;
					// 0, p is inside a shadow volume. Otherwise continue in the negative child
					current = current%4==3 ? 0 : current+1;
		 		}
		 	}
		 	
		}
	}
	return float(current);
}
).";
	
	return { getDpsvVertexShader(), 
		std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fsPrologue + str.str() + fsEpilogue) };
}

std::vector<std::shared_ptr<ge::gl::Shader>> getDpsvHybridProgramShaders(bool enableDepthOptim)
{
	std::stringstream str;
	
	str << R".(
bool DPSV_subQueryStackLess( in vec3 p, uint start, uint stop){ 
	// see DPSV_stackless for detailed comments
	bool secondVisit = false;
	const float dist =  length(p);

	uint current = start;
	while(current!=stop){
		
		const Node n = nodes[ current ];
).";

	if (enableDepthOptim)
	{
		str << R".(
		if (current % 4 == 0 && dist < uintBitsToFloat(n.link[2])) {
			current = nodes[current + 2].link[2];
			secondVisit = true;
		}
		else
).";
	}
	
	str << R".(
		{
			const float offset = dot(n.plane.xyz, p) + n.plane.w;
	
	 		if ( secondVisit==false && n.link[1]>0u && offset*offset / (dist*dist)< uintBitsToFloat(n.link[3])  )
	 			current = n.link[1];
	 		else
	 		{
				if ( offset>=0.0f ){ 
					if (n.link[0]==0){
						secondVisit=true;
						current = nodes[current - current%4 + 2].link[2];
					}
					else{
						secondVisit=false;
						current = n.link[0];
					}
				
				}
				else { 
					secondVisit=false;
					current = (current%4==3) ? 0u : current+1;
					if (current==0) return false;
		 		}
		 	}
		 	
		}
	}
	return true;
}

float traverseToptree( in vec3 p)
{ 
	const uint maxsize = 12 ;
	// (small) stack
    uint stack[maxsize];
    // current stack size
	uint stacksize = 0;
	// distance from p to the light
	const float dist = length(p);

	uint current = root;

	while(current>1)
	{
			// pop
			const Node n = nodes[ current ];
).";

	if (enableDepthOptim)
	{
		str << R".(
			// depth test
			if ( current%4==0 && dist < uintBitsToFloat(n.link[2]) ){   
				current = stacksize > 0 ? stack[ --stacksize ] : 1;
				continue;
			}
).";
	}

	str << R".(		
			const float offset = dot(n.plane.xyz, p) + n.plane.w;

			// wedge test
		 	if ( n.link[1]>0  &&   offset*offset / (dist*dist) < uintBitsToFloat(n.link[3] )  ) 
			   	if (stacksize<maxsize) stack[ stacksize++ ] = n.link[1];
				else{
					// Full stack ! Switch to stackless mode to visit immediately this intersection subtree
					if ( DPSV_subQueryStackLess(p, n.link[1], current)==false ){ 
						current = 0;
						continue;
					}
				}
			
			// positive case
			if ( offset>=0.0 ){	
				if (n.link[0]==0)
					current = stacksize>0 ? stack[ --stacksize ] : 1;					
				else
					current = n.link[0] ;
			}
			else// negative case
				current = ( current%4==3 ) ? 0 : current+1;	
			
	}
	return float(current);
}
).";
	
	return { getDpsvVertexShader(),
		std::make_shared<ge::gl::Shader>(GL_FRAGMENT_SHADER, fsPrologue + str.str() + fsEpilogue) };
}
