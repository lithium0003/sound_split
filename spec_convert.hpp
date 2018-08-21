#pragma once

#include "converter.hpp"
#include "wave.hpp"
#include "analyse.hpp"

enum class WaveChannel {
	Mono,
	Left,
	Right,
};

class SpecConverter: public Converter
{
	protected:
		const WaveData &wav;
	private:
		int callcount = 0;
	protected:
		DetectionUnits detector;

		void convert(std::vector<float> &spec, std::vector<float> &aasc, int slot, int offset, int reqlen) override;
		void filter(const std::vector<float> &spec, std::vector<float> &fspec) override;
	public:
		SpecConverter(const WaveData &wav, WaveChannel channel, double minfq = 20, double maxfq = 22000);
		~SpecConverter() override;
};
