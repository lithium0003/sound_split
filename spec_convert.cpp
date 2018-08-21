#include "spec_convert.hpp"

SpecConverter::SpecConverter(const WaveData &wav, WaveChannel channel, double minfq, double maxfq)
	: Converter(), wav(wav), detector(wav.samplerate, minfq, maxfq) 
{

	samplefq = wav.samplerate;
	maxsamples = wav.loaded_samples; 
	spec_max.resize(ceil((double)maxsamples/samplefq/spec_call_sec), -1);
	fspec_max.resize(ceil((double)maxsamples/samplefq/spec_call_sec), -1);
	fm = detector.fm;
	omh = detector.omh;
	switch(channel){
		case(WaveChannel::Mono):
			data = wav.mono;
			datap = wav.monop;
			break;
		case(WaveChannel::Left):
			data = wav.left;
			datap = wav.leftp;
			break;
		case(WaveChannel::Right):
			data = wav.right;
			datap = wav.rightp;
			break;
	}
}

SpecConverter::~SpecConverter()
{
	while(callcount > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void SpecConverter::convert(std::vector<float> &spec, std::vector<float> &aasc, int slot, int offset, int reqlen)
{
	callcount++;
	detector.AnalyzeData(spec, aasc, offset, reqlen, data, maxsamples, datap);
	callcount--;
}

void SpecConverter::filter(const std::vector<float> &spec, std::vector<float> &fspec)
{
	callcount++;
	detector.FilterData(spec, fspec);
	callcount--;
}
