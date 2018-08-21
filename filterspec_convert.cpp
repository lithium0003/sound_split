#include "filterspec_convert.hpp"
#include "findphase.hpp"

#include <regex>
#include <queue>
#include <utility>
#include <numeric>

FilterSpecConverter::FilterSpecConverter(const WaveData &wav, WaveChannel channel, double minfq, double maxfq)
	: SpecConverter(wav, channel, minfq, maxfq), wav_conv(wav), usechannel(channel)
{
	switch(channel){
		case(WaveChannel::Mono):
			data = wav_conv.mono;
			datap = wav_conv.monop;
			break;
		case(WaveChannel::Left):
			data = wav_conv.left;
			datap = wav_conv.leftp;
			break;
		case(WaveChannel::Right):
			data = wav_conv.right;
			datap = wav_conv.rightp;
			break;
	}
	isrunning = false;
}


std::vector<FilterSpecConverter::cmd> FilterSpecConverter::GetCommands(std::string command_str)
{
	std::stringstream infile(command_str);
	std::vector<struct cmd> commands;

	int startidx = -1;
	int endidx = -1;
	std::vector<std::pair<int,double> > seed;
	int harmonic = -1;
	double freq1 = -1;
	double freq2 = -1;
	for(std::string line; std::getline(infile, line); ){
		std::smatch match;
		std::regex re1(R"(^start\s*(\d+)\s*(\([^)]*\))?\s+(\d+\.?\d*)\s*(Hz)?\s*(@\s*(\d+))?\s*)");
		std::regex re2(R"(^end\s*(\d+)\s*(\([^)]*\))?\s+(\d+\.?\d*)\s*(Hz)?\s*)");
		std::regex re3(R"(^seed\s*(\d+)\s*(\([^)]*\))?\s+(\d+\.?\d*)\s*(Hz)?\s*)");

		if(regex_match(line, match, re1)){
			std::stringstream(match[1].str()) >> startidx;
			std::stringstream(match[3].str()) >> freq1;
			std::stringstream(match[6].str()) >> harmonic;
		}
		else if(regex_match(line, match, re3)){
			int si;
			double fq;
			std::stringstream(match[1].str()) >> si;
			std::stringstream(match[3].str()) >> fq;
			seed.push_back(std::make_pair(si, fq));
		}
		else if(regex_match(line, match, re2)){
			std::stringstream(match[1].str()) >> endidx;
			std::stringstream(match[3].str()) >> freq2;
			if(startidx >= 0 && endidx > startidx && freq1 >= 0 && freq2 >= 0){
				if(endidx > maxsamples) endidx = maxsamples;
				using std::swap;
				if(freq1 > freq2) swap(freq1, freq2);
				if(harmonic < 1) harmonic = 1;
				struct cmd c = {startidx, endidx, freq1, freq2, harmonic, seed};
				if(c.seed.empty()){
					printf("freq %f - %f Hz @ %d overtone\n\tstart %d end %d\n", c.minfreq, c.maxfreq, c.harmonic, c.startidx, c.endidx);
					printf("\tno seed\n");
					commands.push_back(c);
				}
				else {
					bool fail = false;
					for(const auto &s: c.seed){
						if(s.first < c.startidx || s.first >= c.endidx || s.second < c.minfreq || s.second > c.maxfreq){
							fail = true;
							break;
						}
					}
					if(!fail){
						printf("freq %f - %f Hz @ %d overtone\n\tstart %d end %d\n", c.minfreq, c.maxfreq, c.harmonic, c.startidx, c.endidx);
						for(const auto &s: c.seed)
							printf("\tseed %f Hz %d\n", s.second, s.first);
						commands.push_back(c);
					}
				}
			}
			startidx = -1;
			endidx = -1;
			seed.clear();
			harmonic = -1;
			freq1 = -1;
			freq2 = -1;
		}
	}
	return commands;
}

void FilterSpecConverter::work_thread(std::vector<struct cmd> commands)
{
	isrunning = true;
	{
		std::lock_guard<std::mutex> lock(mtx_fspec);

		fspec_map.clear();
		int slot_count = ceil((double)maxsamples/samplefq/spec_call_sec);
		for(int i = 0; i < slot_count; i++)
		{
			fspec_max[i] = -1;
		}
	}
	{
		std::lock_guard<std::mutex> lock(mtx_spec);

		spec_map.clear();
		aasc_map.clear();
		int slot_count = ceil((double)maxsamples/samplefq/spec_call_sec);
		for(int i = 0; i < slot_count; i++)
		{
			spec_max[i] = -1;
		}

		switch(usechannel){
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
	std::sort(commands.begin(), commands.end(), [](const struct cmd &a, const struct cmd &b){ return a.endidx - a.startidx < b.endidx - b.startidx; });
	std::sort(commands.begin(), commands.end(), [](const struct cmd &a, const struct cmd &b){ return a.startidx < b.startidx; });

	const int su = ceil(log10(1+2.0/detector.t)/log10(detector.f)/6);
	printf("Detector T=%d, df=%f, search unit area=%d\n", detector.t, detector.f, su);

	std::vector<float> output(maxsamples);
	if(isinverse){
		memcpy(output.data(), data, maxsamples*sizeof(float));
	}

	int spec_span = spec_call_sec * samplefq;
	for(const auto &c: commands){
		double freq1 = c.minfreq;
		double freq2 = c.maxfreq;
		int harmonic = c.harmonic;
		std::vector<int> unitid;
		for(int i = 0; i < fm; i++){
			unitid.push_back(i);
		}
		std::sort(unitid.begin(), unitid.end(), [this, freq1](const int &a, const int &b) { return fabs(freq1 - omh[a]) < fabs(freq1 - omh[b]); });
		int unitmin = unitid[0];
		std::sort(unitid.begin(), unitid.end(), [this, freq2](const int &a, const int &b) { return fabs(freq2 - omh[a]) < fabs(freq2 - omh[b]); });
		int unitmax = unitid[0];
		int unitlmin = MAX(0, unitmin-1);
		int unitlmax = MIN(fm-1, unitmax+1);

		const int len0 = c.endidx - c.startidx;

		auto seedpoints = c.seed;
		if(seedpoints.empty())
			seedpoints.push_back(std::make_pair((c.startidx + c.endidx)/2, (freq1+freq2)/2));

		std::sort(seedpoints.begin(), seedpoints.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) { return a.first < b.first; });

		seedpoints.insert(seedpoints.begin(), std::make_pair(c.startidx, -1));
		seedpoints.push_back(std::make_pair(c.endidx, -1));

		std::vector<int> baseunit(len0, -1);
		for(int h = 0; h < harmonic; h++){
			printf("%d overtone search\n", h+1);
				
			std::vector<int> peakunit(len0, -1);
			std::vector<float> peakamp(len0);

			for(auto seed = seedpoints.begin()+1; seed != seedpoints.end()-1; ++seed){
				int seedpoint = seed->first;
				if(seedpoint < c.startidx || seedpoint >= c.endidx){
					printf("seedpoint out of time range\n");
					continue;
				}

				double freq3 = seed->second;
				if(freq3 < freq1 || freq3 > freq2){
					printf("seedpoint out of freq range\n");
					continue;
				}
				std::sort(unitid.begin(), unitid.end(), [this, freq3](const int &a, const int &b) { return fabs(freq3 - omh[a]) < fabs(freq3 - omh[b]); });
				int unitseed = unitid[0];

				printf("seed idx %d unit %d\n", seedpoint, unitseed);

				float maxpeak = -1;
				{
					int st = seedpoint;
					int t_idx = seedpoint - c.startidx;
					int punit = (h == 0)? unitseed: baseunit[t_idx];

					int len1 = (seed+1)->first - seedpoint;
					if((seed+1)->second > 0){
						len1 /= 2;
					}
					while(len1 > 0){
						//printf("1:len %d\n", len1);
						int slot = st / spec_span;
						int offset = st % spec_span;

						double stime = slot * spec_call_sec;
						while(!IsLoadReady(stime, false)){
							if(!is_alive) return;
							std::this_thread::sleep_for(std::chrono::milliseconds(250));
						}
						if(!is_alive) return;

						{
							std::lock_guard<std::mutex> lock(mtx_spec);
							auto& target_spec = spec_map[slot];

							int len2 = (offset + len1 > target_spec.size()/fm)? target_spec.size()/fm - offset: len1;
							for(int t=offset; t < len2+offset; t++, t_idx++){
								if(h == 0){
									if(baseunit[t_idx] >= 0)
										goto finish1;
								}
								else{
									punit = baseunit[t_idx];
									if(punit >= 0){
										float freq4 = omh[punit] * (h+1);
										std::sort(unitid.begin(), unitid.end(), 
												[this, freq4](const int &a, const int &b) { return fabs(freq4 - omh[a]) < fabs(freq4 - omh[b]); });
										punit = unitid[0];
									}
								}
								if(punit < 0) continue;
								if(punit < unitmin || punit > unitmax) continue;

								std::vector<std::pair<int, float> >  tamp;
								for(int u2 = MAX(unitlmin, punit - su); u2 <= MIN(unitlmax, punit + su); u2++){
									tamp.push_back(std::make_pair(u2, target_spec[t*fm+u2]));
								}
								std::sort(tamp.begin(), tamp.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b){ return a.second > b.second; });
							
								if(h == 0){	
									if(tamp[0].first < unitmin){ // out of low freq limit
										goto finish1;
									}
									if(tamp[0].first > unitmax){ // out of high freq limit
										goto finish1;
									}

									if(maxpeak < tamp[0].second)
										maxpeak = tamp[0].second;

									if(tamp[0].second < maxpeak * 0.1){ //<-20dB of maxpeak
										goto finish1;
									}

									punit = tamp[0].first;
									baseunit[t_idx] = punit;
								}
								else{
									if(tamp[0].first < unitmin){ // out of low freq limit
										continue;
									}
									if(tamp[0].first > unitmax){ // out of high freq limit
										continue;
									}

									if(tamp[0].second < 0.001){ // <-80dB of max range
										//continue;
									}
								}

								peakunit[t_idx] = tamp[0].first;
								peakamp[t_idx] = tamp[0].second;
							}
							len1 -= len2;
							st += len2;
						}
					}
				}
finish1:
				{
					int st = seedpoint - 1;
					int t_idx = seedpoint - c.startidx - 1;
					int punit = (h == 0)? unitseed: baseunit[t_idx];

					int len1 = seedpoint - 1 - (seed-1)->first;
					if((seed-1)->second > 0){
						len1 /= 2;
					}
					while(len1 > 0){
						//printf("2:len %d\n", len1);
						int slot = st / spec_span;
						int offset = st % spec_span;

						double stime = slot * spec_call_sec;
						while(!IsLoadReady(stime, false)){
							if(!is_alive) return;
							std::this_thread::sleep_for(std::chrono::milliseconds(250));
						}
						if(!is_alive) return;

						{
							std::lock_guard<std::mutex> lock(mtx_spec);
							auto& target_spec = spec_map[slot];

							int len2 = (offset + 1 < len1)? offset+1: len1;
							for(int t=offset; t > offset-len2; t--, t_idx--){
								if(h == 0){
									if(baseunit[t_idx] >= 0)
										goto finish2;
								}
								else{
									punit = baseunit[t_idx];
									if(punit >= 0){
										float freq4 = omh[punit] * (h+1);
										std::sort(unitid.begin(), unitid.end(), 
												[this, freq4](const int &a, const int &b) { return fabs(freq4 - omh[a]) < fabs(freq4 - omh[b]); });
										punit = unitid[0];
									}
								}
								if(punit < 0) continue;
								if(punit < unitmin || punit > unitmax) continue;

								std::vector<std::pair<int, float> >  tamp;
								for(int u2 = MAX(unitlmin, punit - su); u2 <= MIN(unitlmax, punit + su); u2++){
									tamp.push_back(std::make_pair(u2, target_spec[t*fm+u2]));
								}
								std::sort(tamp.begin(), tamp.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b){ return a.second > b.second; });
							
								if(h == 0){	
									if(tamp[0].first < unitmin){ // out of low freq limit
										goto finish2;
									}
									if(tamp[0].first > unitmax){ // out of high freq limit
										goto finish2;
									}

									if(maxpeak < tamp[0].second)
										maxpeak = tamp[0].second;

									if(tamp[0].second < maxpeak * 0.1){ //<-20dB of maxpeak
										goto finish2;
									}

								
									punit = tamp[0].first;
									baseunit[t_idx] = punit;
								}
								else{
									if(tamp[0].first < unitmin){ // out of low freq limit
										continue;
									}
									if(tamp[0].first > unitmax){ // out of high freq limit
										continue;
									}

									if(tamp[0].second < 0.001){ // <-80dB of max range
										//continue;
									}
								}


								peakunit[t_idx] = tamp[0].first;;
								peakamp[t_idx] = tamp[0].second;
							}
							len1 -= len2;
							st -= len2;
						}
					}
				}
finish2:
				;
			}
			const double minanalyse = 0.0025;
			const double maxsilence = 0.001;
			{
				auto p = peakunit.begin();
				std::vector<std::pair<int, int> > cont; // idx, count
				while(p != peakunit.end()){
					if(*p < 0){
						p = std::find_if(p, peakunit.end(), [p](const int &a){ return a != *p; });
					}
					else{
						auto p2 = std::find_if(p, peakunit.end(), [p](const int &a){ return a != *p; });
						cont.push_back(std::make_pair(p - peakunit.begin(), p2 - p));
						p = p2;
					}
				}

				std::vector<int> comb;
				auto cp = cont.begin();
				while(cp != cont.end()){
					if((double)cp->second/samplefq > minanalyse){
						comb.push_back(peakunit[cp->first]);
						++cp;
						continue;
					}
					else{
						auto cp2 = cp;
						if(cp != cont.begin()){
							--cp;
							if((double)cp->second/samplefq > minanalyse){
								comb.push_back(comb.back());
								cp = cp2;
								++cp;
								continue;
							}
							cp = cp2;
						}
						while((double)(std::accumulate(cp2, cp, 0, [](int init, std::pair<int, int> a){ return init+a.second; }))/samplefq <= minanalyse)
							if(++cp == cont.end()) break;

						if(cp == cont.end()) --cp;
						int sumcount = std::accumulate(cp2, cp, 0, [](int init, std::pair<int, int> a){ return init+a.second; });
						int newunit = std::accumulate(cp2, cp, 0, [peakunit](int init, std::pair<int, int> a){ return init+peakunit[a.first]*a.second; })/sumcount;
						while(cp2 != cp){
							comb.push_back(newunit);
							++cp2;
						}
						comb.push_back(newunit);
						cp = cp2;
						++cp;
						continue;
					}
				}

				assert(comb.size() == cont.size());

				for(int i = 0; i < comb.size(); i++){
					assert(cont[i].first+cont[i].second <= len0);
					for(int j = cont[i].first; j < MIN(cont[i].first+cont[i].second, len0); j++)
						peakunit[j] = comb[i];
				}
			}

			const int ulen = unitmax - unitmin + 1;
			std::vector<float> tmpouta(ulen*len0, 0);
			std::vector<float> tmpoutp(ulen*len0, 0);
			std::vector<float> tmpoutf(ulen*len0, -1);

			std::queue<std::thread> waitlist;
			const int maxwait = std::thread::hardware_concurrency();
			for(int u = 0;  u < ulen; u++){
				//printf("unit %d\n", u);
				std::vector<float> tspec(len0, 0);
				for(int i = 0; i < len0; i++){
					if(peakunit[i] == u+unitmin)
						tspec[i] = peakamp[i];
				}

				std::vector<int> silence(len0, 0);
				std::vector<int> sound(len0, 0);
				{
					auto p = tspec.rbegin();
					while(p != tspec.rend()){
						auto p2 = std::find_if(p, tspec.rend(), [](const float a){ return a > 0; });
						int slen = p2 - p;
						for(; p != p2; ++p){
							silence[len0 - (p - tspec.rbegin()) -1] = slen;
						}
						p = std::find_if(p, tspec.rend(), [](const float a){ return a == 0; });
					}	
				}
				{
					auto p = tspec.rbegin();
					while(p != tspec.rend()){
						auto p2 = std::find_if(p, tspec.rend(), [](const float a){ return a == 0; });
						for(int s = 0; p != p2; ++p, s++){
							sound[len0 - (p - tspec.rbegin()) -1] = s;
						}
						p = std::find_if(p, tspec.rend(), [](const float a){ return a > 0; });
					}	
				}

				int st = 0;
				std::vector<float> tmpamp;
				for(int k = 0; k < len0; k++){
					float s = tspec[k];
					if((s > 0 && ((double)tmpamp.size()/samplefq <= minanalyse || (double)sound[k]/samplefq <= minanalyse)) ||
						(s == 0 && (double)silence[k]/samplefq < maxsilence))
					{
						if(tmpamp.empty()) st = k;
						tmpamp.push_back(s);
					}
					else{
						if(s > 0){
							if(tmpamp.empty()) st = k;
							tmpamp.push_back(s);
						}
						if((double)tmpamp.size()/samplefq >= minanalyse){
							waitlist.push(std::thread([=, &tmpouta, &tmpoutp, &tmpoutf]{
								//printf("start %d, len %lu\n", st+c.startidx, tmpamp.size());
								float fq = omh[u+unitmin];
								float ph = FindPhase(fq, samplefq, tmpamp, &data[st+c.startidx]);
								for(int j = 0; j < tmpamp.size(); j++){
									tmpoutp[(st+j)*ulen+u] = 2.0*M_PI*j * fq / samplefq + ph;
									tmpouta[(st+j)*ulen+u] = tmpamp[j];
									tmpoutf[(st+j)*ulen+u] = fq;
								}
							}));
							while(waitlist.size() > maxwait){
								waitlist.front().join();
								waitlist.pop();
							}
						}
						else if(!tmpamp.empty()){
							printf("discard len:%lu start %d unit %d(%d)\n",tmpamp.size(), st+c.startidx, u, u+unitmin);
						}
						tmpamp.clear();
					}
				}
				if(!tmpamp.empty()){
					waitlist.push(std::thread([=, &tmpouta, &tmpoutp, &tmpoutf]{
						//printf("start %d, len %lu\n", st+c.startidx, tmpamp.size());
						float fq = omh[u+unitmin];
						float ph = FindPhase(fq, samplefq, tmpamp, &data[st+c.startidx]);
						for(int j = 0; j < tmpamp.size(); j++){
							tmpoutp[(st+j)*ulen+u] = 2.0*M_PI*j * fq / samplefq + ph;
							tmpouta[(st+j)*ulen+u] = tmpamp[j];
							tmpoutf[(st+j)*ulen+u] = fq;
						}
					}));
					while(waitlist.size() > maxwait){
						waitlist.front().join();
						waitlist.pop();
					}
				}
			}
			while(!waitlist.empty()){
				waitlist.front().join();
				waitlist.pop();
			}

			std::vector<float> sumout(len0);
			double prevph = 0;
			for(int i = 0; i < len0; i++){
				//printf("t %d\n",i);
				double ph = 0;
				double amp1 = 0;
				double amp = 0;
				double f = 0;
				for(int u = 0; u < ulen; u++){
					if(tmpouta[i*ulen+u] > 0 && tmpoutf[i*ulen+u] > 0){
						//printf("unit %d, %f Hz, amp %f, phase %f\n", u, tmpoutf[i*ulen+u], tmpouta[i*ulen+u], tmpoutp[i*ulen+u]);
						ph = fmodf(tmpoutp[i*ulen+u], 2*M_PI);
						f = tmpoutf[i*ulen+u];
						if(amp1 < tmpouta[i*ulen+u]){
							amp1 = tmpouta[i*ulen+u];
						}
						amp += tmpouta[i*ulen+u];
					}
				}
				prevph += 2*M_PI*f/samplefq;
				prevph = fmodf(prevph, 2*M_PI);
				double diff = ph - prevph;
				if(diff < 0) diff += 2*M_PI;
				if(fabs(diff) > M_PI) diff -= 2*M_PI;
				//printf("diff %f deg\n", diff/M_PI*180);
				prevph += std::copysign(MIN(fabs(diff)/10, 0.1*M_PI/180), diff);

				sumout[i] = amp * sin(prevph);
			}

			if(isinverse){
				for(int i = c.startidx, j = 0; i < c.endidx; i++, j++){
					output[i] -= sumout[j];
				}
			}
			else{
				for(int i = c.startidx, j = 0; i < c.endidx; i++, j++){
					output[i] += sumout[j];
				}
			}
		}
	}	

	wav_conv.LoadFrom(output);
	{
		std::lock_guard<std::mutex> lock(mtx_fspec);

		fspec_map.clear();
		int slot_count = ceil((double)maxsamples/samplefq/spec_call_sec);
		for(int i = 0; i < slot_count; i++)
		{
			fspec_max[i] = -1;
		}
	}
	{
		std::lock_guard<std::mutex> lock(mtx_spec);

		spec_map.clear();
		aasc_map.clear();
		int slot_count = ceil((double)maxsamples/samplefq/spec_call_sec);
		for(int i = 0; i < slot_count; i++)
		{
			spec_max[i] = -1;
		}

		switch(usechannel){
			case(WaveChannel::Mono):
				data = wav_conv.mono;
				datap = wav_conv.monop;
				break;
			case(WaveChannel::Left):
				data = wav_conv.left;
				datap = wav_conv.leftp;
				break;
			case(WaveChannel::Right):
				data = wav_conv.right;
				datap = wav_conv.rightp;
				break;
		}
	}
	isrunning = false;
}

void FilterSpecConverter::ApplyFilter(std::string command_str)
{
	if(!isrunning){
		isrunning = true;
		isinverse = false;
		std::vector<struct cmd> commands = GetCommands(command_str);
		if(commands.empty()){
			isrunning = false;
		       	return;
		}

		auto t = std::thread(&FilterSpecConverter::work_thread, this, commands);
		t.detach();
	}
}

void FilterSpecConverter::ApplyInverseFilter(std::string command_str)
{
	if(!isrunning){
		isrunning = true;
		isinverse = true;
		std::vector<struct cmd> commands = GetCommands(command_str);
		if(commands.empty()){
			isrunning = false;
		       	return;
		}

		auto t = std::thread(&FilterSpecConverter::work_thread, this, commands);
		t.detach();
	}
}
