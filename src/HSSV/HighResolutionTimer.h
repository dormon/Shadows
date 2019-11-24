#pragma once

#include <chrono>

/*
 * High Resolution Timer
 */

constexpr double TIMER_NS_PER_SECOND = 1000000000.0f;
constexpr double TIMER_NS_PER_MILLISECOND = 1000000.0f;

class HighResolutionTimer
{
private:
	std::chrono::high_resolution_clock::time_point _start, _stop, _last;
	std::chrono::duration<double, std::nano> _duration;

public:
	HighResolutionTimer();
	void reset();
	double getElapsedTimeSeconds();
	double getElapsedTimeMilliseconds();
	double getElapsedTimeFromLastQuerySeconds();
	double getElapsedTimeFromLastQueryMilliseconds();
	double getTimerPrecisionSeconds();
};