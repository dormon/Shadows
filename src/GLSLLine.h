#pragma once

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define GLSL_LINE "#line " STRINGIZE(__LINE__) "\n"
