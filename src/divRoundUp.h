#pragma once

#include <cstddef>
#include <glm/glm.hpp>

template<typename T, typename D = T, typename std::enable_if_t<std::is_integral<T>::value && std::is_integral<T>::value, char> = 0>
decltype(std::declval<T>()+std::declval<D>()) divRoundUp(T const&dividend, D const&divisor) {
	if (dividend%divisor != 0)
		return (dividend / divisor) + 1;
	return dividend / divisor;
}

inline glm::uvec2 divRoundUp(glm::uvec2 const&divident, glm::uvec2 const&divisor) {
	glm::uvec2 result;
	for (uint32_t i = 0; i < 2; ++i)
		result[i] = static_cast<uint32_t>(divRoundUp(divident[i], divisor[i]));
	return result;
}