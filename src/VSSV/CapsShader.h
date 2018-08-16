#pragma once

#include <string>

std::string const vertexShaderSrc = R".(
#line 6
layout(location=0)in vec3 triangleVertexA;
layout(location=1)in vec3 triangleVertexB;
layout(location=2)in vec3 triangleVertexC;

uniform vec4 lightPosition    = vec4(10,10,10,1);
uniform mat4 modelMatrix      = mat4(1.f);
uniform mat4 viewMatrix       = mat4(1.f);
uniform mat4 projectionMatrix = mat4(1.f);

void main(){
  int multiplicity = currentMultiplicity(triangleVertexA,triangleVertexB,triangleVertexC,lightPosition);
  if(multiplicity==0){gl_Position=vec4(0,0,0,1);return;}
  if(multiplicity>0){ // this has to be swapped if using strips
    float farCap = float(gl_InstanceID&1);
    int   vID    = ((farCap>0)?2-int(gl_VertexID):int(gl_VertexID));

    vec4 P=vec4(triangleVertexB*float(vID==0)+triangleVertexA*float(vID==1)+triangleVertexC*float(vID==2),1.0);
  
    P=P*(1.0-farCap)+vec4(P.xyz*lightPosition.w-lightPosition.xyz,0)*farCap;
    P=projectionMatrix*viewMatrix*modelMatrix*P;
    P=P.xyww*(1.0-farCap)+P*farCap;//front cap
    gl_Position=P;

  }else{

    float farCap = float(gl_InstanceID&1);
    int   vID    = ((farCap>0)?2-int(gl_VertexID):int(gl_VertexID));

    vec4 P=vec4(triangleVertexA*float(vID==0)+triangleVertexB*float(vID==1)+triangleVertexC*float(vID==2),1.0);
  
    P=P*(1.0-farCap)+vec4(P.xyz*lightPosition.w-lightPosition.xyz,0)*farCap;
    P=projectionMatrix*viewMatrix*modelMatrix*P;
    P=P.xyww*(1.0-farCap)+P*farCap;//front cap
    gl_Position=P;
  }
}
).";
