#include <RSSV/collisionShader.h>

std::string const rssv::collisionShader = R".(

const uint TRIVIAL_REJECT = 0xf0u;
const uint TRIVIAL_ACCEPT =    3u;
const uint INTERSECTS     =    2u;

vec4 movePointToSubfrustum(
    in vec4 point    ,
    in vec2 leftRight,
    in vec2 bottomTop,
    in vec2 nearFar  ){
  vec4 result;
  result.x = (point.x + -point.w * ( -1 + leftRight.y + leftRight.x)) / (leftRight.y - leftRight.x);
  result.y = (point.y + -point.w * ( -1 + bottomTop.y + bottomTop.x)) / (bottomTop.y - bottomTop.x);
  result.z = (point.w * (nearFar.y + nearFar.x) - 2 * nearFar.x * nearFar.y ) / (nearFar.y - nearFar.x);
  result.w = point.w;
  return result;
}

// line inresects
// i in {x,y,z}
// X(t) = A+t*(B-A)
// -X(t)w <= X(t)i <= + X(t)w
// 
// -X(t)w + 2aiX(t)w <= X(t)i <= -X(t) + 2biX(t)w
// X(t)w*(-1+2ai) <= X(t)i <= X(t)w*(-1+2bi)
// X(t)w*aai <= X(t)i <= X(t)w*bbi
//
// X(t)w*aai <= X(t)i
// X(t)i     <= X(t)w*bbi
//
// [Aw+t(Bw-Aw)]*aai <= Ai+t(Bi-Ai)
// Ai+t(Bi-Ai)       <= [Aw+t(Bw-Aw)]*bbi
//
// Aw*aai+t(Bw-Aw)aai <= Ai+t(Bi-Ai)
// Ai+t(Bi-Ai)        <= Aw*bbi+t(Bw-Aw)bbi
//
// Aw*aai-Ai          <= t(Bi-Ai)-t(Bw-Aw)aai
// Ai-Aw*bbi          <= t(Bw-Aw)bbi-t(Bi-Ai)
//
// Aw*aai-Ai          <= t[Bi-Ai-(Bw-Aw)aai]
// Ai-Aw*bbi          <= t[(Bw-Aw)bbi-Bi+Ai]
//
// +Aw*aai-Ai          <= t[+Bi-Ai-(Bw-Aw)aai]
// -Aw*bbi+Ai          <= t[-Bi+Ai+(Bw-Aw)bbi]
// M                  <= t*N
// N>0: M/N <= t
// N<0: M/N >= t
// N=0: stop when M>0

bool doesLineInterectSubFrustum(in vec4 A,in vec4 B,in vec3 minCorner,in vec3 maxCorner){
  float tt[2] = {0.f,1.f};
  float M;
  float N;
  uint doMin;

  //#define MINIMIZE()\
  //M/=N;\
  //doMin = uint(N<0.f);\
  //N=(tt[doMin]-M)*(-1.f+2.f*doMin);\
  //tt[doMin] = float(N<0)*tt[doMin] + float(N>=0)*M

  #define MINIMIZE()\
  if(N == 0.f && M >= 0.f)return false;\
  if(N > 0.f)tt[0] = max(tt[0],M/N);\
  if(N < 0.f)tt[1] = min(tt[1],M/N)
  
  M = +A.w*minCorner[0]-A[0];
  N = +B[0]-A[0]-(B.w-A.w)*minCorner[0];
  MINIMIZE();
  M = +A.w*minCorner[1]-A[1];
  N = +B[1]-A[1]-(B.w-A.w)*minCorner[1];
  MINIMIZE();
  M = +A.w*minCorner[2]-A[2];
  N = +B[2]-A[2]-(B.w-A.w)*minCorner[2];
  MINIMIZE();

  M = -A.w*maxCorner[0]+A[0];
  N = -B[0]+A[0]+(B.w-A.w)*maxCorner[0];
  MINIMIZE();
  M = -A.w*maxCorner[1]+A[1];
  N = -B[1]+A[1]+(B.w-A.w)*maxCorner[1];
  MINIMIZE();
  M = -A.w*maxCorner[2]+A[2];
  N = -B[2]+A[2]+(B.w-A.w)*maxCorner[2];
  MINIMIZE();
  
#undef MINIMIZE
  return tt[0] <= tt[1];
}

bool doesSubFrustumDiagonalIntersectSilhouette(in vec3 minCorner,in vec3 maxCorner,in vec4 A,in vec4 B,in vec4 L,in vec4 plane){
  if(sign(plane.x)*sign(plane.z) < 0){
    float z = minCorner.x;
    minCorner.x = maxCorner.x;
    maxCorner.x = z;
  }
  if(sign(plane.y)*sign(plane.z) < 0){
    float z = minCorner.y;
    minCorner.y = maxCorner.y;
    maxCorner.y = z;
  }

  float s = -dot(plane,vec4(minCorner,1))/dot(maxCorner-minCorner,plane.xyz);
  if(s<0)return false;
  if(s>1)return false;

  vec3 D = minCorner+s*(maxCorner-minCorner);

  // X(t,l)                    = A+t(B-A)-lL
  // Di*X(t,l)w                = X(t,l)i
  // Di(Aw+t(Bw-Aw)-lLw)       = Ai+t(Bi-Ai)-lLi
  // DiAw + tDi(Bw-Aw) - lDiLw = Ai + t(Bi-Ai) - lLi
  // DiAw-Ai = t[(Bi-Ai)-Di(Bw-Aw)] + l[DiLw-Li]
  //
  // DxAw-Ax = t[(Bx-Ax)-Dx(Bw-Aw)] + l[DxLw-Lx] 
  // DyAw-Ay = t[(By-Ay)-Dy(Bw-Aw)] + l[DyLw-Ly] 
  //
  // w = tu + lv
  // z = tx + ly

  float u = (B.x-A.x)-(B.w-A.w)*D.x;
  float v = -L.x+L.w*D.x;
  float w = -A.x+D.x*A.w;

  float x = (B.y-A.y)-(B.w-A.w)*D.y;
  float y = -L.y+L.w*D.y;
  float z = -A.y+D.y*A.w;

  float divisor  = u*y - x*v;
  float dividend = w*y - z*v;

  if(divisor == 0.f)return false;
  float t = dividend/divisor;
  if(t < 0.f || t > 1.f)return false;
  if(v == 0.f)return false;
  float l = (w-t*u)/v;
  if(l < 0.f || l > 1.f)return false;

  return true;
}

bool doesSubFrustumDiagonalIntersectTriangle(in vec3 minCorner,in vec3 maxCorner,in vec4 A,in vec4 B,in vec4 C,in vec4 plane){
  if(sign(plane.x)*sign(plane.z) < 0){
    float z = minCorner.x;
    minCorner.x = maxCorner.x;
    maxCorner.x = z;
  }
  if(sign(plane.y)*sign(plane.z) < 0){
    float z = minCorner.y;
    minCorner.y = maxCorner.y;
    maxCorner.y = z;
  }

  float s = -dot(plane,vec4(minCorner,1))/dot(maxCorner-minCorner,plane.xyz);
  if(s<0)return false;
  if(s>1)return false;

  vec3 D = minCorner+s*(maxCorner-minCorner);

  // X(t,l)                         = At+Bl+C(1-t-l)
  // X(t,l)                         = At+Bl+C-tC-lC
  // X(t,l)                         = C+t(A-C)+l(B-C)
  //
  // Di*X(t,l)w                     = X(t,l)i
  // Di(Cw+t(Aw-Cw)+l(Bw-Cw))       = Ci+t(Ai-Ci)+l(Bi-Ci)
  // DiCw + tDi(Aw-Cw) + lDi(Bw-Cw) = Ci + t(Ai-Ci) + l(Bi-Ci)
  // DiCw-Ci                        = t[(Ai-Ci)-Di(Aw-Cw)] + l[(Bi-Ci)-Di(Bw-Cw)]
  //
  // DxCw-Cx                        = t[(Ax-Cx)-Dx(Aw-Cw)] + l[(Bx-Cx)-Dx(Bw-Cw)]
  // DyCw-Cy                        = t[(Ay-Cy)-Dy(Aw-Cw)] + l[(By-Cy)-Dy(Bw-Cw)]
  //
  // w = tu + lv
  // z = tx + ly

  float w = D.x*C.w-C.x;
  float u = (A.x-C.x)-D.x*(A.w-C.w);
  float v = (B.x-C.x)-D.x*(B.w-C.w);

  float z = D.y*C.w-C.y;
  float x = (A.y-C.y)-D.y*(A.w-C.w);
  float y = (B.y-C.y)-D.y*(B.w-C.w);

  float divisor  = u*y - x*v;
  float dividend = w*y - z*v;

  if(divisor == 0.f)return false;
  float t = dividend/divisor;
  if(t < 0.f || t > 1.f)return false;
  if(v == 0.f)return false;
  float l = (w-t*u)/v;
  if(l < 0.f || l > 1.f)return false;
  if(t+l>1)return false;

  return true;
}


#line 110
bool doesEdgeIntersectFrustum(in vec4 A,in vec4 B){
  float tMin = 0.f;                         //register R0
  float tMax = 1.f;                         //register R1
  float divident;
  float divisor;
  #define MINIMIZE()\
  if(divisor < 0.f)tMin = max(tMin,divident/divisor);\
  if(divisor > 0.f)tMax = min(tMax,divident/divisor);\
  if(divisor == 0.f && divident < 0.f)tMin = 2.f

  divident = A[0]+A[3];
  divisor  = divident-B[0]-B[3];
  MINIMIZE();
  divident = A[1]+A[3];
  divisor  = divident-B[1]-B[3];
  MINIMIZE();
  divident = A[2]+A[3];
  divisor  = divident-B[2]-B[3];
  MINIMIZE();
#line 130
  divident = -A[0]+A[3];
  divisor  = divident+B[0]-B[3];
  MINIMIZE();
  divident = -A[1]+A[3];
  divisor  = divident+B[1]-B[3];
  MINIMIZE();
  divident = -A[2]+A[3];
  divisor  = divident+B[2]-B[3];
  MINIMIZE();

  #undef MINIMIZE
  return tMin <= tMax;
}
#line 143
bool doesDiagonalIntersectShadowVolumeSide(in vec4 A,in vec4 B,in vec4 L,in uint d){
  float a = -1.f + 2.f*float(d/2u);
  float b = -1.f + 2.f*float(d&1u);
  float u = B.x - A.x - a*B.y + a*A.y;
  float v = a*L.y - L.x;
  float w = a*A.y - A.x;
  float x = B.x - A.x - b*B.z + b*A.z;
  float y = b*L.z - L.x;
  float z = b*A.z - A.x;
  float divisor  = u*y - x*v;
  float dividend = w*y - z*v;
  if(divisor == 0.f)return false;
  float t = (w*y - z*v) / (u*y - x*v);
  if(t < 0.f || t > 1.f)return false;
  if(v == 0.f)return false;
  float l = (w-t*u)/v;
  if(l < 0.f || l > 1.f)return false;
  vec4 pp = mix(A,B,t)-L*l;
  return all(greaterThanEqual(pp.xyz,-pp.www))&&all(lessThanEqual(pp.xyz,+pp.www));
}

bool doesShadowVolumeSideIntersectsFrustum(in vec4 A,in vec4 B,in vec4 L){
  if(doesEdgeIntersectFrustum(A,A-L))return true;
  if(doesEdgeIntersectFrustum(B,B-L))return true;
  if(doesEdgeIntersectFrustum(A,B))return true;
  if(doesDiagonalIntersectShadowVolumeSide(A,B,L,0))return true;
  if(doesDiagonalIntersectShadowVolumeSide(A,B,L,1))return true;
  if(doesDiagonalIntersectShadowVolumeSide(A,B,L,2))return true;
  if(doesDiagonalIntersectShadowVolumeSide(A,B,L,3))return true;
  return false;
}

uint silhouetteStatus(vec4 edgeA,vec4 edgeB,vec4 light,vec3 minCorner,vec3 maxCorner){
  vec2 leftRight = vec2(minCorner.x,maxCorner.x)*.5+.5;
  vec2 bottomTop = vec2(minCorner.y,maxCorner.y)*.5+.5;
  vec2 nearFar   = vec2(-DEPTH_TO_Z(minCorner.z),-DEPTH_TO_Z(maxCorner.z));

  if(doesLineInterectSubFrustum(edgeA,edgeA-light,minCorner,maxCorner))return INTERSECTS;
  if(doesLineInterectSubFrustum(edgeB,edgeB-light,minCorner,maxCorner))return INTERSECTS;
  if(doesLineInterectSubFrustum(edgeA,edgeB      ,minCorner,maxCorner))return INTERSECTS;

  vec4 A = movePointToSubfrustum(edgeA,leftRight,bottomTop,nearFar);
  vec4 B = movePointToSubfrustum(edgeB,leftRight,bottomTop,nearFar);
  vec4 L = movePointToSubfrustum(light,leftRight,bottomTop,nearFar);

  if(doesShadowVolumeSideIntersectsFrustum(A,B,L))
    return INTERSECTS;

  return TRIVIAL_REJECT;
}

int sampleCollision(
    in vec3 startSample,
    in vec3 endSample  ,
    in vec4 aa         ,
    in vec4 bb         ,
    in vec4 ll         ){

  vec3 edgeNormal     = normalize(cross(bb.xyz*aa.w-aa.xyz*bb.w,ll.xyz*aa.w-aa.xyz*ll.w));
  vec4 edgePlane      = vec4(edgeNormal,-dot(edgeNormal,aa.xyz/aa.w));

  vec3 sampleNormal   = normalize(cross(endSample-startSample,ll.xyz-startSample*ll.w));
  vec4 samplePlane    = vec4(sampleNormal,-dot(sampleNormal,startSample));

  vec3 triangleNormal = normalize(cross(endSample-startSample,bb.xyz*aa.w-aa.xyz*bb.w));
  vec4 trianglePlane  = vec4(triangleNormal,-dot(triangleNormal,startSample));

  // m n c 
  // - - 0
  // 0 - 1
  // + - 1
  // - 0 1
  // 0 0 0
  // + 0 0
  // - + 1
  // 0 + 0
  // + + 0
  //
  // m<0 && n>=0 || m>=0 && n<0
  // m<0 xor n<0

  if((dot(edgePlane,vec4(startSample,1))<0) == (dot(edgePlane,vec4(endSample,1))<0))return 0;

  if(dot(trianglePlane,aa)<=0)return 0;

  return 1-int(sign(dot(samplePlane,aa)*dot(samplePlane,bb)));
}


).";
