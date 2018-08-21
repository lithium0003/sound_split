#pragma once

#include "fmrs.hpp"

float FindPhase(float &cfreq, int samplefq, const std::vector<float> &ampwav, const float *orgwave);
std::vector<float> FindMatchWave(int samplefq, const std::vector<float> &ampwav, const std::vector<float> &cfreq, const float *orgwave);
