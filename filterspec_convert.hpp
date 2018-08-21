#pragma once

#include "spec_convert.hpp"

#include <string>
#include <vector>

class FilterSpecConverter: public SpecConverter
{
	public:
		struct cmd {
			int startidx;
			int endidx;
			double minfreq;
			double maxfreq;
			int harmonic;
			std::vector<std::pair<int, double> > seed;
		};
	protected:
		WaveData wav_conv;
	private:
		bool isrunning;
		bool isinverse;
		WaveChannel usechannel;
		
		void work_thread(std::vector<struct cmd> commands);
		std::vector<struct cmd> GetCommands(std::string command_str);
	public:
		FilterSpecConverter(const WaveData &wav, WaveChannel channel, double minfq = 20, double maxfq = 22000);

		void ApplyFilter(std::string command_str);
		void ApplyInverseFilter(std::string command_str);
};
