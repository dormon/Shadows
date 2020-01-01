#include "MTSV_shaders.h"

std::string getMtsvShadowMaskVs()
{
	return R".(
#version 450 core

void main()
{
	gl_Position = vec4(-1 + 2*(gl_VertexID / 2), -1 + 2*(gl_VertexID % 2), 0, 1);
}
).";
}


std::string getMtsvBuildCsShader()
{
	return std::string(R".(
layout (local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

struct Node
{
	uint compressed_dir; // Direction to the 1st vertex of the capsule, compressed
    float angle; // Angle of the cone/capsule
    float delta; // Partitionning distance
    uint compressed_dir2; // Direction to the 2nd vertex of the capsule, compressed. Unused if the node contains a cone
    uvec4 children; // x : positive child ptr, 
                    // y : intersected child ptr, 
                    // z : negative child ptr, 
                    // w : bounding distance of the intersection area
};

// I store the vertices in Woop's unit space to perform ray/triangle intersection during traversal
// See A Ray Tracing Hardware Architecture for Dynamic Scene
struct Triangle
{
    vec4 a;
    vec4 b;
    vec4 c;
};

struct TriangleBoundingCircle 
{
    vec3 center;
    vec3 reference_point; // which triangle's vertex gave the radius ( used to compute projective distance )
    float radius;
};

layout (std430, binding=0) readonly buffer VERTICES {
    float vertices[];
};

layout (std430, binding=1) writeonly buffer WorldTriangles {
    Triangle world_triangles[];
};

layout (std430, binding=2) buffer TOPTree {
    Node nodes[]; 
};

layout (std430,binding=3) buffer util
{
    uint root; // Initalized to 0
	uint node;          // Used to perform a "nodes[node++] when you add a node to the buffer, but atomically" 
    uint build_count;   // initialized to 0, used for persistant thread scheduling
} index;


vec3 get_vertex(uint index) {
    uint id = index * 8;
    return vec3(vertices[id], vertices[id+1], vertices[id+2]);
}
// ------------------------------------------------------------------------------------------

// Number of triangles to insert in the tree
uniform uint triangle_count = 0;

// Should contain the angle of the cone that bounds the scene as seen from the light
// i.e. the cone with the light as center
// Personally I compute the AABB of the scene on the CPU using the AABBs of all objects,
// Then I compute the bounding cone from it
// If the light is inside the AABB, I set the cone angle to PI/2
uniform float delta = 0.0;

uniform vec3 light_pos;

#define PERSISTANT_THREAD_CONSTRUCT 1
shared uint s_num_triangle[32];


/* Distance to a triangle function by inigo quilez - iq/2013
  https://www.shadertoy.com/view/4sXXRN
  License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
  https://creativecommons.org/licenses/by-nc-sa/3.0/
*/
float dot2(in vec3 v)
{ 
	return dot(v,v); 
}

float tri2LightDistance(in vec3 v1, in vec3 v2, in vec3 v3, in vec3 p)
{
    const vec3 v21 = v2 - v1; vec3 p1 = p - v1;
    const vec3 v32 = v3 - v2; vec3 p2 = p - v2;
    const vec3 v13 = v1 - v3; vec3 p3 = p - v3;
    const vec3 nor = cross(v21, v13);
    
    return sqrt((sign(dot(cross(v21,nor),p1)) + 
                 sign(dot(cross(v32,nor),p2)) + 
                 sign(dot(cross(v13,nor),p3))<2.0) 
                ?
                min(min(
        dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0,1.0)-p1), 
        dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0,1.0)-p2)), 
                    dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0,1.0)-p3))
                :
                dot(nor,p1)*dot(nor,p1)/dot2(nor));
}


// http://realtimecollisiondetection.net/blog/?p=20
TriangleBoundingCircle minimum_bounding_circle( vec3 a, vec3 b, vec3 c )
{
    float dotABAB = dot( b - a, b - a );
    float dotABAC = dot( b - a, c - a );
    float dotACAC = dot( c - a, c - a );
    float d = 2.0f * ( dotABAB*dotACAC - dotABAC*dotABAC );
    
    TriangleBoundingCircle result;
    result.reference_point = a;
    
    if ( abs( d ) <= 10e-5 ) 
    {
        // a, b, and c lie on a line. Circle center is center of AABB of the
        // points, and radius is distance from circle center to AABB corner
        vec3 aabb_min = min( a, min( b, c ) );
        vec3 aabb_max = max( a, max( b, c ) );
        result.center = ( aabb_min + aabb_max ) * 0.5f;
        result.reference_point = aabb_min;
    }
    else
    {
        float s = ( dotABAB*dotACAC - dotACAC*dotABAC ) / d;
        float t = ( dotACAC*dotABAB - dotABAB*dotABAC ) / d;
        // s controls height over AC, t over AB, ( 1-s-t ) over BC
        if ( s <= 0.0f )
        {
            result.center = 0.5f * ( a + c );
        } 
        else if ( t <= 0.0f ) 
        {
            result.center = 0.5f * ( a + b );
        } 
        else if ( s + t >= 1.0f )
        {
            result.center = 0.5f * ( b + c );
            result.reference_point = b;
        } 
        else 
        {
            result.center  = a + s*( b - a ) + t*( c - a );
        }
    }
    
    result.radius = length( result.center - result.reference_point );
    return( result );
}


// Computes the angle between two vectors "centered" at the light
// The angle can be computed with acos(dot(a, b))...
// But when the angle approches zero (which is often the case) 
// the precision of floating points is insuffificent
// That's why I use asin(...) instead
float angular_distance(vec3 a, vec3 b)
{
    float cosine = dot(a, b);
    float angle = (cosine < 0.8) ? acos(cosine) : asin(length(cross(a, b)));
    return(angle);
}


// Unit vector (de)compression code from the magnificent Inigo Quilez
vec2 msign( vec2 v )
{
    return vec2( (v.x>=0.0) ? 1.0 : -1.0, 
                (v.y>=0.0) ? 1.0 : -1.0 );
}

uint octahedral_32( in vec3 nor )
{
    nor /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*msign(nor.xy);
    return packSnorm2x16(nor.xy);
}

vec3 i_octahedral_32( uint data )
{
    vec2 v = unpackSnorm2x16(data);
    vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z,0.0);                     // much faster than original
    nor.x += (nor.x>0.0)?-t:t;                     // implementation of this
    nor.y += (nor.y>0.0)?-t:t;                     // technique
    
    return normalize( nor );
}

float point_segment_angular_distance( vec3 unit_p, vec3 unit_s1, vec3 unit_s2, vec3 unit_pp, float dir_angle ) 
{
    //float a1 = angular_distance(unit_s1, unit_s2);
    float a2 = angular_distance(unit_s1, unit_pp);
    float a3 = angular_distance(unit_s2, unit_pp);
    
    if( a2 < dir_angle && a3 < dir_angle && dot(normalize(unit_s1+unit_s2), unit_pp) > 0.0 ) 
    {
        return angular_distance(unit_p, unit_pp);
    } 
    else
    {
        return min(angular_distance(unit_p, unit_s1), 
                   angular_distance(unit_p, unit_s2));
    }
}

float segment_segment_angular_distance(vec3 d0, vec3 d1, vec3 n0, float dir_angle0, vec3 d2, vec3 d3, vec3 n1, float dir_angle1, out float max_dist)
{
    
    vec3 dpd0 = normalize(d0 - n1 * dot( n1, d0 ));
    vec3 dpd1 = normalize(d1 - n1 * dot(n1, d1));
    
    float dist0 = point_segment_angular_distance(d0, d2, d3, dpd0, dir_angle1);
    float dist1 = point_segment_angular_distance(d1, d2, d3, dpd1, dir_angle1);
    
    max_dist = max(dist0, dist1);
    
    return min(dist0, dist1);
}


void MTree_mergeTriangle(in uint i)
{
    vec3 v0 = get_vertex(id0 + shadow_cmd_vertex_offset[cmd_id]);
    vec3 v1 = get_vertex(id1 + shadow_cmd_vertex_offset[cmd_id]);
    vec3 v2 = get_vertex(id2 + shadow_cmd_vertex_offset[cmd_id]);
    
    // Don't forget to transpose the vertices to world space *but* with the light as origin
    mat4 model_mat = matrices[cmd_id];
    vec3 A = (model_mat * vec4(v0, 1.0)).xyz - light_pos;
    vec3 B = (model_mat * vec4(v1, 1.0)).xyz - light_pos;
    vec3 C = (model_mat * vec4(v2, 1.0)).xyz - light_pos;
    
    // Add the current node to the node buffer atomically  
    const uint insertion = atomicAdd(index.node, 1);
    
    { 
    	// I store the vertices in "Woop's" space for
        // ray/triangle intersection test during traversal
        // http://jcgt.org/published/0002/01/05/paper.pdf
        const vec3 ba = B - A;
        const vec3 ca = C - A;
        const vec3 n2 = cross(ba, ca);
        mat4 mat = mat4(vec4(ba.x, ca.x, n2.x, A.x),
                        vec4(ba.y, ca.y, n2.y, A.y),
                        vec4(ba.z, ca.z, n2.z, A.z),
                        vec4(0.f   , 0.f  , 0.f ,  1.f)) ;
        mat4 matInv = inverse(mat);
        
        // inverse mapping (1,0,0,0) to AB
        vec4 v1 = vec4(matInv[0][0], matInv[0][1], matInv[0][2], matInv[0][3]);
        // inverse mapping [0,1,0,0] to AC
        vec4 v2 = vec4(matInv[1][0], matInv[1][1], matInv[1][2], matInv[1][3]);
        // inverse mapping [0,0,1,0] to N
        vec4 v0 = vec4(matInv[2][0], matInv[2][1], matInv[2][2], matInv[2][3]);
        // where is A? with Charlie? No!
        // in fact it is (0,0,0,0) ... B maps to (1,0,0) and C to (0,1,0), N maps to (0,0,1).
        
        world_triangles[insertion] = Triangle(v0, v1, v2);
    }
    
    // Compute the capping plane equation (i.e. the supporting plane of triangle ABC)
    const vec3 norm = normalize(cross(B-A, C-A));
    const vec4 capping_plane = vec4(norm, -dot(norm,A));
    
    
    // Perform front face culling
    if (capping_plane.w > 0.0) 
	{
        return;
    }
    
#if 1
    // Need when front face culling
    vec3 tmp = A;
    A = C;
    C = tmp;
#endif
    
    Node node;
    
    vec3 center = (A + B + C) / 3.0;
    vec3 cone_dir = normalize(center);
    
    // Normalized directions to vertices
    vec3 nA = normalize(A);
    vec3 nB = normalize(B);
    vec3 nC = normalize(C);
    
    // Angular distance of the triangle edges
    vec3 edge_angles = vec3(
        angular_distance(nA, nB),
        angular_distance(nA, nC),
        angular_distance(nB, nC));
    
    // Compute the id of the largest edge
    uint edge_id = 0;
    edge_id = (edge_angles[0] > edge_angles[1]) ? 0 : 1;
    edge_id = (edge_angles[2] > edge_angles[edge_id]) ? 2 : edge_id;
    
    // Compute id of the smallest edge
    uint edge_id3 = (edge_angles[0] < edge_angles[1]) ? 0 : 1;
    edge_id3 = (edge_angles[2] < edge_angles[edge_id3]) ? 2 : edge_id3;
    
    
    float angle, dir_angle;
    uint compressed_dir, compressed_dir2;
    vec3 normal, d0, d1;
    float diff = 0.0;

    // Choose between capsule and cone
    // If the largest edge is n times larger than the smallest one, use a capsule, otherwise a cone
    // In the clustered version, the choice is made during the preprocess according to the OBB edge lengths
    // Here it is "on the fly", and the value of n is a bit more scene dependent
    // (more capsules is less conservative for thin and elongated triangles but it is also more costely to compute there distances)
    if(edge_angles[edge_id] > 120.0*edge_angles[edge_id3])
	{
        // Compute capsule's angle
        vec3 s0;
        if(edge_id3==0)
		{
            s0 = (nA + nB) * 0.5;
        } 
		else if(edge_id3==1)
		{
            s0 = (nA + nC) * 0.5;
        }
		else 
		{
            s0 = (nB + nC) * 0.5;
        }
        s0 = normalize(s0);
        
        vec3 edge;
        if(edge_id==0)
		{
            edge = nA-nB;
        } 
		else if(edge_id==1)
		{
            edge = nA-nC;
        }
		else
		{
            edge = nB-nC;
        }
		
        vec3 s1 = normalize(s0 - edge);        
        
        vec3 normal = normalize(cross(s0, s1));
        
        dir_angle = angular_distance(s0, s1);     
        
        vec3 dpdA = normalize(nA - normal * dot(normal, nA));
        vec3 dpdB = normalize(nB - normal * dot(normal, nB));
        vec3 dpdC = normalize(nC - normal * dot(normal, nC));
        
        angle = point_segment_angular_distance(nA, s0, s1, dpdA, dir_angle);
        angle = max(angle,
                    point_segment_angular_distance(nB, s0, s1, dpdB, dir_angle));
        angle = max(angle,
                    point_segment_angular_distance(nC, s0, s1, dpdC, dir_angle));
        
        compressed_dir = octahedral_32(s0);
        compressed_dir2 = octahedral_32(s1);
        node.compressed_dir = compressed_dir;
        node.compressed_dir2 = compressed_dir2;
        
        d0 = i_octahedral_32(compressed_dir);
        d1 = i_octahedral_32(compressed_dir2);
        
        // Angle between the two vectors pointing to the capule's vertices
        // Used during the capsule/capsule angular distance computation 
        diff = max(angular_distance(s0, d0), angular_distance(s1, d1));        
    }
	else
	{
        // Compute the cone's angle
        float a1 = angular_distance(cone_dir, nA);
        float a2 = angular_distance(cone_dir, nB);
        float a3 = angular_distance(cone_dir, nC);
        
        const TriangleBoundingCircle bounding_circle = minimum_bounding_circle( nA, nB, nC );
        cone_dir = normalize( bounding_circle.center );
        angle = asin( bounding_circle.radius / length( bounding_circle.center ) );
        
        dir_angle = 0;
        d0 = cone_dir;
        compressed_dir = octahedral_32(cone_dir);
        compressed_dir2 = 0u;
        node.compressed_dir = compressed_dir;
        node.compressed_dir2 = 0u;
    }
 
    node.angle = angle + diff;
    node.delta = delta;
    
    node.children = uvec4(0);
    
    // Min distance from node to sphere as in the 2nd paper from Gerhards on PSV (EGSR16)
    // Not used here, tbh I don't remember if it was better to use it or not in this version
    const uint distance = floatBitsToUint(tri2LightDistance(A, B, C, vec3(0.0)));
    
    nodes[insertion] = node;
    
    
    // Below : insertion loop similar to the PSV algorithm
    uint current = atomicCompSwap(index.root, 0, insertion);
        
    while(current != 0)
	{
        Node father = nodes[current];
        
        float father_delta = father.delta;
        vec4 father_cone = vec4(i_octahedral_32(father.compressed_dir), father_delta);
        
        const vec3 father_d0 = i_octahedral_32(father.compressed_dir);
        const vec3 father_d1 = i_octahedral_32(father.compressed_dir2);
        
        
        // Stores the min_dist between elements
        float dist;

        // Stores the max_dist between elements
        float max1;

        // Used for the capsule/capsule distance computation
        bool visual_intersection = false;
        
        // Below, we compute the distance between the two elements
        // It can be a capsule/capsule distance, a cone/capsule distance, a capsule/cone distance or a cone/cone distance
        // We know that an element is a cone if *compressed_dir2* is set to zero
        const vec3 father_normal = normalize(cross(father_d0, father_d1));
        
        if(compressed_dir2 != 0 && father.compressed_dir2 != 0) 
		{ 
			// Capsule/Capsule angular distance
            const float father_dir_angle = angular_distance(father_d0, father_d1);
            float max0;

            float dist0 =
                segment_segment_angular_distance( father_d0, father_d1, father_normal, father_dir_angle, d0, d1, normal, dir_angle, max0);
            float dist1 =
                segment_segment_angular_distance( d0, d1, normal, dir_angle, father_d0, father_d1, father_normal, father_dir_angle, max1);
            
            dist = min(dist0, dist1);

            
            // If those conditions are true this means that the supporting segments of the capsules intersect visually
            // If so the visual_intersection flag is set to true 
            if(sign(dot(father_normal, d0)) != sign(dot(father_normal, d1))) {
                if(sign(dot(normal, father_d0)) != sign(dot(normal, father_d1))) {
                    visual_intersection = true;
                }
            }

        }
		else if(compressed_dir2 == 0 && father.compressed_dir2 != 0) 
		{ 
			// Capsule/Cone angular distance (capsule as pivot)
            const float father_dir_angle = angular_distance(father_d0, father_d1);
            vec3 dpd0 = normalize(d0 - father_normal * dot( father_normal, d0 ));
            dist = point_segment_angular_distance( d0, father_d0, father_d1, dpd0, father_dir_angle );
            max1 = dist;
        }
).") + R".(
		else if(compressed_dir2 != 0 && father.compressed_dir2 == 0) 
		{
			// Cone/Capsule angular distance (cone as pivot)
            vec3 dpd0 = normalize(father_d0 - normal * dot( normal, father_d0 ));
            dist = point_segment_angular_distance( father_d0, d0, d1, dpd0, dir_angle );
            max1 = max(dist, max(angular_distance(father_d0, d0), angular_distance(father_d0, d1)));
        } 
		else 
		{
			// Cone/Cone distance
            dist = angular_distance( father_d0, d0 );
            max1 = dist;
        }

        float cone_delta_side = 0.0;
        if(max1+angle < father.delta) 
		{
            cone_delta_side = -1.0;
        } 
		else if(dist-angle > father.delta && vi == false) 
		{
            cone_delta_side = 1.0;
        }  
        
        if (cone_delta_side > 0) 
		{ 
			// Current element is in the positive space of its father
            current = atomicCompSwap(nodes[current].children[0], 0, insertion);
        } 
		else if (cone_delta_side < 0) 
		{ 
			// Current element is in the negative space of its father    
        	nodes[insertion].delta *= 0.5;
            current = atomicCompSwap(nodes[current].children[2], 0, insertion);

        } 
		else 
		{
            // Current element intersects the positive and negative space of its father

            // The term wedge is kept from the terminology of the paper of Gerhards et al. on PSV (EG15)
            float in_wedge = (visual_intersection) ? father.delta : father.delta - ( dist - angle );
            float out_wedge = (max1 + angle) - father.delta;

            // Wedge is the max distance of the element "on each side" of the boundary defined by his father and his partitioning distance (delta)
            float wedge = max(in_wedge, out_wedge);
            
			// Update the distance that bounds the region containing intersected geometry for the father            
            atomicMax(nodes[current].children[3], floatBitsToUint(wedge));

            current = atomicCompSwap(nodes[current].children[1], 0, insertion);
        }   
    }
}


void main(void) 
{
    // In here you just want to insert an element in the tree with the function MTree_mergeTriangle
    // For a straightforward implementation you just want the ith thread to insert the ith element,
    // And spawn ceil(triangle_count / work_group_size)+1 threads
    // Here I use persistant threads. 
    // If you have implemented the PSV algorithm of Gerhards, do the same as you already did
	const uint warpSize1 = gl_SubgroupSize - 1;
	const uint dataIndex = gl_LocalInvocationID.x / gl_SubgroupSize;
	
    for( uint k=0 ; k<triangle_count ; k++ ) 
	{     
        if ( ( gl_LocalInvocationID.x & warpSize1) == uint( 0 ) )
		{
            s_num_triangle[dataIndex] = atomicAdd( index.build_count, gl_SubgroupSize );
		}
        
        uint warp_id = s_num_triangle[dataIndex]/gl_SubgroupSize ;
        uint warp_count = uint(ceil(float(triangle_count)/float(gl_SubgroupSize)));

        uint element_id;
        if(warp_id >= warp_count)
		{
            return;
        }
        
#if 1
        element_id = warp_id*gl_SubgroupSize  + (gl_LocalInvocationID.x & warpSize1);
#else
        uint cycle_length = min(warp_count, 128u);
        uint bucket_count = (warp_count) / cycle_length;
        if((gl_LocalInvocationID.x & warpSize1) == 0u)
		{
            if(warp_id/cycle_length == bucket_count) 
			{
                element_id = warp_id * gl_SubgroupSize ;
            } 
			else
			{
                element_id = (warp_id%bucket_count)*cycle_length + warp_id/bucket_count;
                element_id *= gl_SubgroupSize;
            } 
        }
        element_id = readInvocationARB(element_id, 0);//shuffleNV(element_id, 0, gl_SubgroupSize);
        element_id += (gl_LocalInvocationID.x & warpSize1);
#endif 

        if (element_id>=triangle_count) 
		{
			break;
		}
        
        MTree_mergeTriangle(element_id);
    }
}
).";
}

std::string getMtsvShadowMaskFs()
{
    return R".(
#version 450 core

layout( binding = 0 ) uniform sampler2D positionTex; 

uniform vec3 lightPos;

layout(location=0) out float fColor;

struct Node
{
    uint compressed_dir; // Direction to the 1st vertex of the capsule, compressed

    float angle; // Angle of the cone/capsule

    float delta; // Partitionning distance

    uint compressed_dir2; // Direction to the 2nd vertex of the capsule, compressed. Unused if the node contains a cone

    uvec4 children; // x : positive child ptr, 
                    // y : intersected child ptr, 
                    // z : negative child ptr, 
                    // w : bounding distance of the intersection area
};

struct Triangle
{
    vec4 a;
    vec4 b;
    vec4 c;
};

// See compute shader comments
layout (std430, binding=0) buffer TOPTree 
{
    Node nodes[];
};

layout ( std430, binding=1) buffer TOPTreeRoot
{
    uint root; // initalized to 0
};

layout (std430, binding=2) buffer WorldTriangles
{
    Triangle world_triangles[];
};

float angular_distance( vec3 a, vec3 b ) 
{
    float cosine = dot( a, b );
    float angle = ( cosine < 0.8 ) ? acos( cosine ) : asin( length( cross( a, b ) ) );
    return( angle );
}

vec2 msign( vec2 v )
{
    return vec2( (v.x>=0.0) ? 1.0 : -1.0, 
                (v.y>=0.0) ? 1.0 : -1.0 );
}

uint octahedral_32( in vec3 nor )
{
    nor /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*msign(nor.xy);
    
    return packSnorm2x16(nor.xy);
    //uvec2 d = uvec2(round(32767.5 + nor.xy*32767.5));  return d.x|(d.y<<16u);
}

vec3 i_octahedral_32( uint data )
{
    vec2 v = unpackSnorm2x16(data);
    //uvec2 iv = uvec2( data, data>>16u ) & 65535u; vec2 v = vec2(iv)/32767.5 - 1.0;
    
    vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z,0.0);                     // much faster than original
    nor.x += (nor.x>0.0)?-t:t;                     // implementation of this
    nor.y += (nor.y>0.0)?-t:t;                     // technique
    
    return normalize( nor );
}

//#define ANGLE(a, b) acos(dot(a, b))
#define ANGLE(a, b) angular_distance(a, b)
//#define ANGLE(a, b) asin(length(cross(a, b)))
float point_segment_angular_distance( vec3 unit_p, vec3 unit_s1, vec3 unit_s2, vec3 unit_pp, float a1 )
{    
    //float a1 = ANGLE(unit_s1, unit_s2);
    float a2 = ANGLE(unit_s1, unit_pp);
    float a3 = ANGLE(unit_s2, unit_pp);
    
    if( a2 < a1 && a3 < a1 && dot(normalize(unit_s1+unit_s2), unit_pp) > 0.0)
	{
        return ANGLE(unit_p, unit_pp);
    } 
	else
	{
        return min(ANGLE(unit_p, unit_s1), 
                   ANGLE(unit_p, unit_s2));
    }   
}

float MTree_traversal(in vec3 p)
{
    uint stack[32];
    uint stack_size = 1u;
    stack[0] = 0u;
    
    uint current = root;
        
    const float dist = length(p);
    const vec3 point_dir = p / dist;
    
    do 
	{
        const Node node = nodes[current];
        const vec3 node_d0 = i_octahedral_32(node.compressed_dir);
        
        float d;
        
        if(node.compressed_dir2 == 0) 
		{ 
            d = angular_distance(point_dir, node_d0);
        } 
		else
		{
            const vec3 node_d1 = i_octahedral_32(node.compressed_dir2);
            
            vec3 capsule_n = normalize(cross(node_d0, node_d1));
            vec3 dpd0 = normalize(point_dir - capsule_n * dot( capsule_n, point_dir ));
            
            float a = angular_distance(node_d0, node_d1);
            
            d = point_segment_angular_distance(point_dir, node_d0, node_d1, dpd0, a);
        }
        
        if(d < node.angle)
		{
            vec3 rd = point_dir;
            vec3 ro = vec3(0.0);
            Triangle triangle = world_triangles[current];
            const float nd = dot(triangle.a.xyz, rd);
            const float nf = dot(triangle.a, vec4(ro, 1.0));
            
            const float invNd = 1.0 / nd;
            const float tt = - nf * invNd; // d == 0 !
            if(tt > 0 && tt < dist)
			{
                const float Ox = dot(triangle.b, vec4(ro, 1.0));
                const float Dx = dot(triangle.b.xyz, rd);
                const float beta = tt * Dx + Ox;
                if (beta >= 0.f)
				{
                    const float Oy = dot(triangle.c, vec4(ro, 1.0));
                    const float Dy = dot(triangle.c.xyz, rd);
                    const float gamma =  tt * Dy + Oy;
                    if(gamma >= 0.f && beta + gamma <= 1.f)
					{
                        return 0.f;
                    }
                }
            }
        }
        
        if(node.children[1] > 0 
           //&& (d > node.delta - uintBitsToFloat(node.children[3]) && (d < node.delta + uintBitsToFloat(node.out_wedge)))
           && (abs(d-node.delta) < (uintBitsToFloat(node.children[3])))
           ) 
		{
            stack[stack_size++] = node.children[1];
        }
        
        if(d > node.delta) 
		{
            current = ( node.children[0] > 0 ) ? node.children[0] : stack[--stack_size];
        }
		else
		{
            current = ( node.children[2] > 0 ) ? node.children[2] : stack[--stack_size];
        }
        
    } while(current > 0);
    
    return 1.f;
}

void main()
{
    const vec4 pos = texelFetch( positionTex, ivec2(gl_FragCoord.xy), 0 );

    if(pos.w == 0) 
	{
        fColor = 1.f;
        return;
    }
     
    fColor = MTree_traversal(pos.xyz-lightPos);  
}
).";
}
