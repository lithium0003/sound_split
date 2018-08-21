#include "converter.hpp"
#include "colormap.h"

Converter::Converter()
	: pixbuf(NULL), data(NULL), datap(NULL), convert_done(NULL)
{
	show_t_sec = 5.0;
	show_s_sec = 0;
	is_alive = true;

	thread = std::thread(&Converter::spec_thread, this);
	thread2 = std::thread(&Converter::filter_thread, this);
}

Converter::~Converter()
{
	printf("Converter shutdown...\n");
	is_alive = false;
	{
		std::lock_guard<std::mutex> lock(mtx_analyse);
		wait_analyse.push_back(0);
		cv_analyse.notify_all();
	}
	{
		std::lock_guard<std::mutex> lock(mtx_filter);
		wait_filter.push_back(0);
		cv_filter.notify_all();
	}
	thread.join();
	thread2.join();
}

double mainlobe(double Hz, double peakHz, double peakValue)
{
	double f = log10(Hz) - log10(peakHz);
	double a = 0*log10(Hz) - 1e4;
	return a*f*f + peakValue+0.1;
}

double lowerslope(double Hz, double peakHz, double b)
{
	double f1 = log10(fabs(Hz-peakHz)/peakHz);
	double f2 = log10(Hz);
	return -17*f1 + 14*f2 + b;
}

double higherslope(double Hz, double peakHz, double b)
{
	double f1 = log10(fabs(Hz-peakHz)/peakHz);
	double f2 = log10(Hz);
	return -19*f1 + 22*f2 + b;
}

std::vector<float> Converter::findmask(double Hz, double value)
{
	std::vector<float> mask(fm);

	double valuedB = 20*log10(value);
	double lowerPoint = Hz - Hz*0.05;
	double higherPoint = Hz + Hz*0.05;
	double b1 = mainlobe(lowerPoint, Hz, valuedB) - lowerslope(lowerPoint, Hz, 0);
	double b2 = mainlobe(higherPoint, Hz, valuedB) - higherslope(higherPoint, Hz, 0);

	for(int i = 0; i < fm; i++){
		double fq = omh[i];
		double value;
		if(fq < Hz*1.002 && fq > Hz*0.998)
			value = valuedB;
		else if(fq < lowerPoint)
			value = lowerslope(fq, Hz, b1);
		else if(fq > higherPoint)
			value = higherslope(fq, Hz, b2);
		else
			value = mainlobe(fq, Hz, valuedB);
		mask[i] = pow(10, value/20);
	}

	return mask;
}

void Converter::filter_thread()
{
	int callcount = 0;
	while(is_alive){
		int slot_idx;
		{
			std::unique_lock<std::mutex> lock(mtx_filter);
			cv_filter.wait(lock, [this]{ return !wait_filter.empty(); });
			slot_idx = wait_filter.front();
			wait_filter.pop_front();
		}
		if(!is_alive) break;
		
		int spec_span = spec_call_sec * samplefq;
		int offset = slot_idx * spec_span;
		int reqlen = (offset + spec_call_sec*samplefq < maxsamples)? spec_call_sec*samplefq: maxsamples - offset;
		if(reqlen < 0 || offset < 0)
			continue;

		{
			std::lock_guard<std::mutex> lock(mtx_fspec);
			if(!fspec_map[slot_idx].empty())
				continue;
		}
		if(!is_alive) break;


		while(!SpecReady(slot_idx)){
			if(!is_alive) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
		}
		if(!is_alive) break;

		const int protect = 2;
		if(fspec_slotcount > 3+protect){
			std::lock_guard<std::mutex> lock(mtx_fspec);
			int minslot = (show_s_sec - show_t_sec)/spec_call_sec;
			int maxslot = (show_s_sec + show_t_sec*3)/spec_call_sec;
			int slot_count = ceil((double)maxsamples/samplefq/spec_call_sec);

			for(int i = protect; i < slot_count; i++)
			{
				if(i == converting_slot) continue;
				if(fspec_max[i] < 0) continue;
				if(i >= minslot && i <= maxslot) continue;

				fspec_map.erase(i);
				fspec_max[i] = -1;
			}
		}

		std::vector<float> *target_spec;
		std::vector<float> *target_fspec;
		{
			std::lock_guard<std::mutex> lock(mtx_spec);
			target_spec = &spec_map[slot_idx];
		}
		{
			std::lock_guard<std::mutex> lock(mtx_fspec);
			target_fspec = &fspec_map[slot_idx];
		}
		if(!is_alive) break;
		filter(*target_spec, *target_fspec);
		if(!is_alive) break;
		callcount++;
		std::thread subt([this,target_fspec, slot_idx, &callcount]{
				float m = 0;
				for(const auto s: *target_fspec){
					if(s > m) m = s;
				}
				fspec_max[slot_idx] = m;
				fspec_slotcount++;
				if(filter_done)
					filter_done(slot_idx);
				callcount--;
		});
		subt.detach();
	}
	while(callcount > 0){
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

void Converter::spec_thread()
{
	int callcount = 0;
	while(is_alive){
		int slot_idx;
		{
			std::unique_lock<std::mutex> lock(mtx_analyse);
			cv_analyse.wait(lock, [this]{ return !wait_analyse.empty(); });
			slot_idx = wait_analyse.front();
			wait_analyse.pop_front();
		}
		if(!is_alive) break;
		
		int spec_span = spec_call_sec * samplefq;
		int offset = slot_idx * spec_span;
		int reqlen = (offset + spec_call_sec*samplefq < maxsamples)? spec_call_sec*samplefq: maxsamples - offset;
		if(reqlen < 0 || offset < 0)
			continue;
		double st = (double)offset / samplefq;
		double et = (double)(offset+reqlen) / samplefq;
		if(et < show_s_sec - show_t_sec * 2 || st > show_s_sec + show_t_sec * 5)
			continue;

		{
			std::lock_guard<std::mutex> lock(mtx_spec);
			if(!spec_map[slot_idx].empty())
				continue;
		}
		if(!is_alive) break;
		const int protect = 2;
		if(spec_slotcount > 3+protect){
			std::lock_guard<std::mutex> lock(mtx_spec);
			int minslot = (show_s_sec - show_t_sec)/spec_call_sec;
			int maxslot = (show_s_sec + show_t_sec*3)/spec_call_sec;
			int slot_count = ceil((double)maxsamples/samplefq/spec_call_sec);

			for(int i = protect; i < slot_count; i++)
			{
				if(i == converting_slot) continue;
				if(spec_max[i] < 0) continue;
				if(i >= minslot && i <= maxslot) continue;

				spec_map.erase(i);
				aasc_map.erase(i);
				spec_max[i] = -1;
			}
		}

		std::vector<float> *target_spec;
		std::vector<float> *target_aasc;
		{
			std::lock_guard<std::mutex> lock(mtx_spec);
			target_spec = &spec_map[slot_idx];
			target_aasc = &aasc_map[slot_idx];
		}
		if(!is_alive) break;
		convert(*target_spec, *target_aasc, slot_idx, offset, reqlen);
		if(!is_alive) break;
		callcount++;
		std::thread subt([this,target_spec, slot_idx, &callcount]{
				float m = 0;
				for(const auto s: *target_spec){
					if(s > m) m = s;
				}
				spec_max[slot_idx] = m;
				spec_slotcount++;
				if(convert_done)
					convert_done(slot_idx);
				callcount--;
		});
		subt.detach();
	}
	while(callcount > 0){
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

bool Converter::LoadData(double show_t_sec, double show_s_sec)
{
	this->show_t_sec = show_t_sec;
	this->show_s_sec = show_s_sec;
	fill_spec();

	return IsReady(show_s_sec);
}

void Converter::print_HzdB(cairo_t *cr, int idx, double Hz, double value, const struct drawarea &area, double y_range, double y_min)
{
	int top = area.top;
	int left = area.left;
	int height = area.height;
	int width = area.width;

	int x = (double)idx / fm * (width - ax_out)+left+ax_out;
	int y = height - ax_out + top - (log10(value) - y_min) / y_range * (height - ax_out);

	cairo_set_source_rgb(cr, 1, 1, 1);
	cairo_arc(cr, x, y, 1, 0, 2*M_PI);
	cairo_fill(cr);

	std::stringstream out;
	out << std::fixed << std::setprecision(1) << Hz << "Hz " << 20*log10(value) << "dB";
	std::string tag = out.str();

	cairo_text_extents_t extents;
	cairo_text_extents(cr, tag.c_str(), &extents);

	cairo_move_to(cr, x+5, y - extents.y_bearing/2);
	cairo_show_text(cr, tag.c_str());
}

bool Converter::DrawSpecAdd(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max, bool main)
{

	int top = area.top;
	int left = area.left;
	int height = area.height;
	int width = area.width;

	double y_range = y_max - y_min;
	y_min /= 20;
	y_range /= 20;

	int offset = target_sec * samplefq;
	int spec_span = spec_call_sec * samplefq;
	
	bool ret = true;
	if(offset >= 0 && offset < maxsamples){
		std::vector<float> spec1(fm);
		if(IsFilterRun){
			std::unique_lock<std::mutex> lock(mtx_fspec, std::try_to_lock);
			if(lock.owns_lock()){
				int slot = offset / spec_span;
				int t3 = offset % spec_span;
				if(!fspec_map[slot].empty() && (t3+1)*fm-1 < (int)fspec_map[slot].size()){
					for(int u = 0; u < fm; u++){
						spec1[u] = fspec_map[slot][t3*fm + u];
					}
				}
				else{
					ret = false;
				}
			}
		}
		else{
			std::unique_lock<std::mutex> lock(mtx_spec, std::try_to_lock);
			if(lock.owns_lock()){
				int slot = offset / spec_span;
				int t3 = offset % spec_span;
				if(!spec_map[slot].empty() && (t3+1)*fm-1 < (int)spec_map[slot].size()){
					for(int u = 0; u < fm; u++){
						spec1[u] = spec_map[slot][t3*fm + u];
					}
				}
				else{
					ret = false;
				}
			}
		}

		if(main)
			cairo_set_source_rgb(cr, 0, 1, 0);
		else
			cairo_set_source_rgb(cr, 0, 0, 1);
		cairo_set_line_width(cr, 1);
		cairo_move_to(cr, left+ax_out, top+height-ax_out);
		for(int x = left+ax_out; x < left+width; x++){
			int f = (double)(x-ax_out-left) / (width-ax_out) * fm;
			float value = spec1[f];
			if(value > 0)
				value = log10(value);
			else
				value = y_min;
			if(value < y_min)
				value = y_min;
			int y = height - ax_out + top - (value - y_min) / y_range * (height - ax_out);
			cairo_line_to(cr, x, y);
		}
		cairo_stroke(cr);

		std::vector<float> mask(fm);
		int count = fm;
		while(is_alive && count-- > 0){
			double max_s = 0;
			int max_idx = -1;
			double max_Hz = 0;
			const double cutoff = 5000;
			const double noiselevel = pow(10, -50.0/20);
			for(int u = 0; u < fm; u++){
				if(spec1[u] <= mask[u]) continue;
				if(omh[u] > cutoff) break;
				if(spec1[u] < noiselevel) continue;
				if(max_s < spec1[u]){
					max_idx = u;
					max_s = spec1[u];
				}
			}
			if(max_s > 0 && max_idx >= 0){
				max_Hz = omh[max_idx];
				print_HzdB(cr, max_idx, max_Hz, max_s, area, y_range, y_min);
				auto mask1 = findmask(max_Hz, max_s);
				for(int i=0; i < fm; i++){
					//if(mask[i] < mask1[i])
					//	mask[i] = mask1[i];
					mask[i] += mask1[i];
				}
				if(mask[max_idx] < max_s)
					mask[max_idx] = max_s;
				if(max_idx > 0 && mask[max_idx-1] < max_s) 
					mask[max_idx-1] = max_s;
				if(max_idx < fm-1 && mask[max_idx+1] < max_s) 
					mask[max_idx+1] = max_s;
			}
			else{
				break;
			}
		}

		if(main)
			cairo_set_source_rgba(cr, 0, 1, 0, 0.2);
		else
			cairo_set_source_rgba(cr, 0, 0, 1, 0.2);
		cairo_set_line_width(cr, 1);
		cairo_move_to(cr, left+ax_out, top+height-ax_out);
		for(int x = left+ax_out; x < left+width; x++){
			int f = (double)(x-ax_out-left) / (width-ax_out) * fm;
			float value = mask[f];
			if(value > 0)
				value = log10(value);
			else
				value = y_min;
			if(value < y_min)
				value = y_min;
			int y = height - ax_out + top - (value - y_min) / y_range * (height - ax_out);
			cairo_line_to(cr, x, y);
		}
		cairo_line_to(cr, left+width, top+height-ax_out);
		cairo_close_path(cr);
		cairo_fill(cr);
	}
	return ret;
}

bool Converter::DrawSpec(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max, bool main)
{
	int top = area.top;
	int left = area.left;
	int height = area.height;
	int width = area.width;

	cairo_set_source_rgb(cr, 0, 0, 0);
	cairo_rectangle(cr, left, top, width, height);
	cairo_fill(cr);

	bool ret = DrawSpecAdd(cr, area, target_sec, y_min, y_max, main);

	double y_range = y_max - y_min;
	y_min /= 20;
	y_range /= 20;

	cairo_set_source_rgb(cr, 1, 1, 1);
	cairo_set_line_width(cr, 1);

	cairo_move_to(cr, left+ax_out-1, top);
	cairo_line_to(cr, left+ax_out-1, top+height-ax_out+1);
	cairo_line_to(cr, left+width, top+height-ax_out+1);

	const int tic_l = 10;
	for(double v = 0; v <= 10; v++){
		int y = height - ax_out + top - v * 0.1 * (height - ax_out);
		cairo_move_to(cr, left+ax_out-tic_l, top+y);
		cairo_line_to(cr, left+ax_out-1, top+y);

		std::stringstream out;
		out << std::fixed << std::setprecision(1) << 20 * (v * 0.1 * y_range + y_min);
		std::string tag = out.str() + "dB";

		cairo_text_extents_t extents;
		cairo_text_extents(cr, tag.c_str(), &extents);

		cairo_move_to(cr, left, top + y - extents.y_bearing/2);
		cairo_show_text(cr, tag.c_str());
	}
	
	std::list<int> xtics;
	for(int Hz = 20; Hz < 100; Hz += 10)
		xtics.push_back(Hz);
	for(int Hz = 100; Hz < 1000; Hz += 100)
		if(Hz != 900)
			xtics.push_back(Hz);
	for(int Hz = 1000; Hz < 10000; Hz += 1000)
		if(Hz <= 5000 || Hz % 2000 == 0) 
			xtics.push_back(Hz);
	for(int Hz = 10000; Hz <= 20000; Hz += 10000)
		xtics.push_back(Hz);

	for(int x = left+ax_out; x < width+left; x++){
		int f1 = (double)(x-ax_out-left) / (width-ax_out) * fm;
		if(f1 < 0) f1 = 0;
		if(f1 >= fm) f1 = fm-1;
		while(!xtics.empty() && omh[f1] >= xtics.front()){
			cairo_move_to(cr, left+x, top+height-ax_out+1);
			cairo_line_to(cr, left+x, top+height-ax_out+tic_l);

			cairo_text_extents_t extents;
			cairo_text_extents(cr, std::to_string(xtics.front()).c_str(), &extents);
			
			cairo_move_to(cr, left+x-extents.width/2, top+height-ax_out+tic_l+extents.height);
			cairo_show_text(cr, std::to_string(xtics.front()).c_str());
			xtics.pop_front();
		}
		if(xtics.empty()) break;
	}
	cairo_stroke(cr);


	cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
	cairo_set_line_width(cr, 0.5);

	std::list<std::pair<double, std::string> > pitch;
	std::list<std::pair<double, int> > key;
	std::string note[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
	bool dur[] = {true, false, true, false, true, true, false, true, false, true, false, true};
	bool moll[] = {true, false, true, true, false, true, false, true, true, false, true, false};
	for(int i = 0; i < 88; i++){
		int h = (i + 9) / 12;
		int n = (i + 9) % 12;
		std::ostringstream out;
		out << note[n] << h;
		double Hz = A440 * pow(2, (i - 48)/12.0);
		pitch.push_back(std::make_pair(Hz, out.str()));
		key.push_back(std::make_pair(Hz, 9+i));
	}

	for(int x = left+ax_out; x < width+left; x++){
		int f1 = (double)(x-ax_out-left) / (width-ax_out) * fm;
		if(f1 < 0) f1 = 0;
		if(f1 >= fm) f1 = fm-1;
		while(!pitch.empty() && omh[f1] >= pitch.front().first){
			if(tonality == Tonality::None){
				if(pitch.front().second.rfind("#") != std::string::npos){
					cairo_set_source_rgb(cr, 0.5, 0.5, 0.5);
					cairo_set_line_width(cr, 0.5);
				}
				else if(pitch.front().second.find("C") != std::string::npos){
					cairo_set_source_rgb(cr, 1, 1, 1);
					cairo_set_line_width(cr, 1);
				}
				else{
					cairo_set_source_rgb(cr, 0.8, 0.8, 0.8);
					cairo_set_line_width(cr, 0.5);
				}
			}
			else if(static_cast<int>(tonality) < static_cast<int>(Tonality::as_Moll)) {
				// Dur
				int k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::C_Dur))*7 + 12*12) % 12;
				if(k == 0){
					cairo_set_source_rgb(cr, 1, 0, 0);
					cairo_set_line_width(cr, 1);
				}
				else if(dur[k]){
					cairo_set_source_rgb(cr, 0.8, 0.7, 0);
					cairo_set_line_width(cr, 0.5);
				}
				else{
					cairo_set_source_rgb(cr, 0.5, 0.5, 0.5);
					cairo_set_line_width(cr, 0.5);
				}
			}
			else{
				// Moll
				int k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::c_Moll))*7 + 12*12) % 12;
				if(k == 0){
					cairo_set_source_rgb(cr, 1, 0, 0);
					cairo_set_line_width(cr, 1);
				}
				else if(moll[k]){
					cairo_set_source_rgb(cr, 0.8, 0.7, 0);
					cairo_set_line_width(cr, 0.5);
				}
				else{
					cairo_set_source_rgb(cr, 0.5, 0.5, 0.5);
					cairo_set_line_width(cr, 0.5);
				}
			}
			cairo_move_to(cr, x, top);
			cairo_line_to(cr, x, top+height-ax_out);
			cairo_stroke(cr);
			
			if(tonality == Tonality::None){
				if(pitch.front().second.rfind("#") == std::string::npos){
					cairo_text_extents_t extents;
					cairo_text_extents(cr, pitch.front().second.c_str(), &extents);

					cairo_set_source_rgb(cr, 1, 1, 1);
					cairo_move_to(cr, x - extents.width/2, top + height - ax_out - extents.height);
					cairo_show_text(cr, pitch.front().second.c_str());
					cairo_stroke(cr);
				}
			}
			else {
				int k;
				bool dispon;
				if(static_cast<int>(tonality) < static_cast<int>(Tonality::as_Moll)){
					k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::C_Dur))*7 + 12*12) % 12;
					dispon = dur[k];
				}
				else{
					k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::c_Moll))*7 + 12*12) % 12;
					dispon = moll[k];
				}
				if(dispon){
					cairo_text_extents_t extents;
					cairo_text_extents(cr, pitch.front().second.c_str(), &extents);

					cairo_set_source_rgb(cr, 1, 1, 1);
					cairo_move_to(cr, x - extents.width/2, top + height - ax_out - extents.height);
					cairo_show_text(cr, pitch.front().second.c_str());
					cairo_stroke(cr);
				}
			}
			pitch.pop_front();
			key.pop_front();
		}
		if(pitch.empty()) break;
	}


	return ret;
}

bool Converter::DrawImage(cairo_t *cr, const struct drawarea &area, double show_t_sec, double show_s_sec, double play_t, double select_fq, double min_freq, double max_freq)
{
	LoadData(show_t_sec, show_s_sec);

	int top = area.top;
	int left = area.left;
	int height = area.height;
	int width = area.width;

	int fm_min = 0;
	if(min_freq >= 0){
		while(fm_min < fm && omh[fm_min] < min_freq)
			fm_min++;
	}
	int fm_count = fm - fm_min;
	if(max_freq >= 0){
		int fm_max = fm-1;
		while(fm_max > 0 && omh[fm_max] > max_freq)
			fm_max--;
		if(fm_max > fm_min)
			fm_count = fm_max - fm_min + 1;
		else
			fm_count = 1;
	}


	if(!pixbuf || width != gdk_pixbuf_get_width(pixbuf) || height != gdk_pixbuf_get_height(pixbuf)){
		pixbuf = gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8, width, height);
		
		int rowstride = gdk_pixbuf_get_rowstride(pixbuf);
		int n_channels = gdk_pixbuf_get_n_channels(pixbuf);
		guchar *pixels = gdk_pixbuf_get_pixels(pixbuf);
		memset(pixels, 0, rowstride*height);
		for(int x = ax_out; x < width; x++){
			for(int y = 0; y < height-ax_out; y++){
				guchar *p = pixels + y * rowstride + x * n_channels;
				p[0] = parula[0][0] * 255;
				p[1] = parula[0][1] * 255;
				p[2] = parula[0][2] * 255;
			}
		}
	}

	int spec_span = spec_call_sec * samplefq;

	int rowstride = gdk_pixbuf_get_rowstride(pixbuf);
	int n_channels = gdk_pixbuf_get_n_channels(pixbuf);
	guchar *pixels = gdk_pixbuf_get_pixels(pixbuf);

	bool notyet = false;
	if(IsFilterRun){
		std::unique_lock<std::mutex> lock(mtx_fspec, std::try_to_lock);
		if(lock.owns_lock()){
			std::unique_lock<std::mutex> lock2(mtx_spec, std::try_to_lock);
			if(lock2.owns_lock()){
				for(int x = ax_out; x < width; x++){
					double t1 = (double)(x-ax_out) / (width-ax_out) * show_t_sec + show_s_sec;
					int t2 = t1 * samplefq;
					int t3 = t2 % spec_span;
					int slot = t2 / spec_span;

					for(int y = 0; y < height-ax_out; y++){
						guchar *p = pixels + y * rowstride + x * n_channels;
						int f = (double)(height - ax_out - y - 1) / (height-ax_out) * fm_count + fm_min;
						double value = 0;
						if(fspec_max[slot] >= 0 && !fspec_map[slot].empty() && t3*fm+f < (int)fspec_map[slot].size()){
							value = fspec_map[slot][t3*fm + f];
							if(value > 0){
								value = 20 * log10(value);
								if(value < -70) value = 0;
								else{
									value -= -70;
									value = value / 70 * 255;
								}
							}
							if(value > 255) value = 255;
						}
						else if (spec_max[slot] > 0 && !spec_map[slot].empty() && t3*fm+f < (int)spec_map[slot].size()){
							value = spec_map[slot][t3*fm + f];
							value = value / spec_max[slot] * 255;
							if(value > 255) value = 255;
						}
						else{
							notyet = true;
						}
						p[0] = parula[(int)value][0] * 255;
						p[1] = parula[(int)value][1] * 255;
						p[2] = parula[(int)value][2] * 255;
					}
				}
			}
		}
	}
	else{
		std::unique_lock<std::mutex> lock(mtx_spec, std::try_to_lock);
		if(lock.owns_lock()){
			for(int x = ax_out; x < width; x++){
				double t1 = (double)(x-ax_out) / (width-ax_out) * show_t_sec + show_s_sec;
				int t2 = t1 * samplefq;
				int t3 = t2 % spec_span;
				int slot = t2 / spec_span;
				double m = spec_max[slot];

				for(int y = 0; y < height-ax_out; y++){
					guchar *p = pixels + y * rowstride + x * n_channels;
					int f = (double)(height - ax_out - y - 1) / (height-ax_out) * fm_count + fm_min;
					double value = 0;
					if(m > 0 && !spec_map[slot].empty() && t3*fm+f < (int)spec_map[slot].size()){
						value = spec_map[slot][t3*fm + f];
						value = value / m * 255;
						if(value > 255) value = 255;
					}
					else{
						notyet = true;
					}
					p[0] = parula[(int)value][0] * 255;
					p[1] = parula[(int)value][1] * 255;
					p[2] = parula[(int)value][2] * 255;
				}
			}
		}
	}


	gdk_cairo_set_source_pixbuf(cr, pixbuf, left, top);
	cairo_paint(cr);

	cairo_set_source_rgb(cr, 1, 1, 1);
	cairo_set_line_width(cr, 1);

	cairo_move_to(cr, left+ax_out-1, top);
	cairo_line_to(cr, left+ax_out-1, top+height-ax_out+1);
	cairo_line_to(cr, left+width, top+height-ax_out+1);

	const int tic_l = 10;
	std::list<int> xtics;
	for(int Hz = 20; Hz < 100; Hz += 10)
		xtics.push_back(Hz);
	for(int Hz = 100; Hz < 1000; Hz += 100)
		xtics.push_back(Hz);
	for(int Hz = 1000; Hz < 10000; Hz += 1000)
		xtics.push_back(Hz);
	for(int Hz = 10000; Hz <= 20000; Hz += 10000)
		xtics.push_back(Hz);

	if(fm_min > 0){
		int f1 = fm_min;
		while(!xtics.empty() && omh[f1] >= xtics.front()){
			xtics.pop_front();
		}
	}
	for(int y = height-ax_out; y >= 0; y--){
		int f1 = (double)(height - ax_out - y) / (height-ax_out) * fm_count + 0.5 + fm_min;
		if(f1 < 0) f1 = 0;
		if(f1 >= fm) f1 = fm-1;
		while(!xtics.empty() && omh[f1] >= xtics.front()){
			cairo_move_to(cr, left+ax_out-tic_l, top+y);
			cairo_line_to(cr, left+ax_out-1, top+y);
			
			cairo_text_extents_t extents;
			cairo_text_extents(cr, std::to_string(xtics.front()).c_str(), &extents);
			
			cairo_move_to(cr, left, top + y - extents.y_bearing/2);
			cairo_show_text(cr, std::to_string(xtics.front()).c_str());
			xtics.pop_front();
		}
		if(xtics.empty()) break;
	}
	double t2 = floor(show_s_sec * 10) / 10;
	for(int x = ax_out+left; x < width+left; x++){
		double t1 = (double)(x-ax_out-left) / (width-ax_out) * show_t_sec + show_s_sec;
		if(t2 < t1){
			t2 = floor(t1 * 10 + 1)/10;
			cairo_move_to(cr, left+x, top+height-ax_out+1);
			cairo_line_to(cr, left+x, top+height-ax_out+tic_l);

			std::ostringstream out;
			out << std::fixed << std::setprecision(1) << t1;
			cairo_text_extents_t extents;
			cairo_text_extents(cr, out.str().c_str(), &extents);

			cairo_move_to(cr, left+x-extents.width/2, top+height-ax_out+tic_l+extents.height);
			cairo_show_text(cr, out.str().c_str());

			while((t2 - t1) / show_t_sec * (width-ax_out) < extents.width*2){
				t2 = floor(t2 * 10 + 1)/10;
			}
		}
	}

	cairo_stroke(cr);

	cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
	cairo_set_line_width(cr, 0.5);

	std::list<std::pair<double, std::string> > pitch;
	std::list<std::pair<double, int> > key;
	std::string note[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
	bool dur[] = {true, false, true, false, true, true, false, true, false, true, false, true};
	bool moll[] = {true, false, true, true, false, true, false, true, true, false, true, false};
	for(int i = 0; i < 88; i++){
		int h = (i + 9) / 12;
		int n = (i + 9) % 12;
		std::ostringstream out;
		out << note[n] << h;
		double Hz = A440 * pow(2, (i - 48)/12.0);
		pitch.push_back(std::make_pair(Hz, out.str()));
		key.push_back(std::make_pair(Hz, 9+i));
	}

	if(fm_min > 0){
		int f1 = fm_min;
		while(!pitch.empty() && omh[f1] >= pitch.front().first){
			pitch.pop_front();
		}
	}
	for(int y = height-ax_out; y >= 0; y--){
		int f1 = (double)(height - ax_out - y) / (height-ax_out) * fm_count + 0.5 + fm_min;
		if(f1 < 0) f1 = 0;
		if(f1 >= fm) f1 = fm-1;
		while(!pitch.empty() && omh[f1] >= pitch.front().first){
			if(tonality == Tonality::None){
				if(pitch.front().second.rfind("#") != std::string::npos){
					cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
					cairo_set_line_width(cr, 0.5);
				}
				else if(pitch.front().second.find("C") != std::string::npos){
					cairo_set_source_rgb(cr, 0, 0, 0);
					cairo_set_line_width(cr, 1);
				}
				else{
					cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
					cairo_set_line_width(cr, 0.5);
				}
			}
			else if(static_cast<int>(tonality) < static_cast<int>(Tonality::as_Moll)) {
				// Dur
				int k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::C_Dur))*7 + 12*12) % 12;
				if(k == 0){
					cairo_set_source_rgb(cr, 1, 0, 0);
					cairo_set_line_width(cr, 1);
				}
				else if(dur[k]){
					cairo_set_source_rgb(cr, 0, 0, 0);
					cairo_set_line_width(cr, 0.5);
				}
				else{
					cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
					cairo_set_line_width(cr, 0.5);
				}
			}
			else{
				// Moll
				int k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::c_Moll))*7 + 12*12) % 12;
				if(k == 0){
					cairo_set_source_rgb(cr, 1, 0, 0);
					cairo_set_line_width(cr, 1);
				}
				else if(moll[k]){
					cairo_set_source_rgb(cr, 0, 0, 0);
					cairo_set_line_width(cr, 0.5);
				}
				else{
					cairo_set_source_rgb(cr, 0.2, 0.2, 0.2);
					cairo_set_line_width(cr, 0.5);
				}
			}
			cairo_move_to(cr, left+ax_out, top+y);
			cairo_line_to(cr, left+width, top+y);
			cairo_stroke(cr);
			
			if(tonality == Tonality::None){
				if(pitch.front().second.rfind("#") == std::string::npos){
					cairo_text_extents_t extents;
					cairo_text_extents(cr, pitch.front().second.c_str(), &extents);

					cairo_set_source_rgb(cr, 1, 1, 1);
					cairo_move_to(cr, left+ax_out, top + y - extents.y_bearing/2);
					cairo_show_text(cr, pitch.front().second.c_str());
					cairo_stroke(cr);
				}
			}
			else {
				int k;
				bool dispon;
				if(static_cast<int>(tonality) < static_cast<int>(Tonality::as_Moll)){
					k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::C_Dur))*7 + 12*12) % 12;
					dispon = dur[k];
				}
				else{
					k = (key.front().second - (static_cast<int>(tonality) - static_cast<int>(Tonality::c_Moll))*7 + 12*12) % 12;
					dispon = moll[k];
				}
				if(dispon){
					cairo_text_extents_t extents;
					cairo_text_extents(cr, pitch.front().second.c_str(), &extents);

					cairo_set_source_rgb(cr, 1, 1, 1);
					cairo_move_to(cr, left+ax_out, top + y - extents.y_bearing/2);
					cairo_show_text(cr, pitch.front().second.c_str());
					cairo_stroke(cr);
				}
			}
			pitch.pop_front();
			key.pop_front();
		}
		if(pitch.empty()) break;
	}

	if(select_fq > 0){
		int f1 = 0;
		while(f1 < fm && omh[f1] < select_fq)
			f1++;
		int yp = height - ax_out - (double)(f1 - fm_min + 0.5)/fm_count * (height-ax_out);
		if(yp >= 0 && yp < height-ax_out){
			cairo_set_source_rgb(cr, 0, 1, 0);
			cairo_set_line_width(cr, 1);
			cairo_move_to(cr, left+ax_out+1, top+yp);
			cairo_line_to(cr, left+width, top+yp);
			cairo_stroke(cr);

			std::ostringstream out;
			out << std::fixed << std::setprecision(1) << select_fq << "Hz Unit(" << f1 << ")";
			cairo_text_extents_t extents;
			cairo_text_extents(cr, out.str().c_str(), &extents);

			cairo_set_source_rgb(cr, 1, 1, 1);
			cairo_move_to(cr, left+ax_out+20, top+yp-extents.height/2);
			cairo_show_text(cr, out.str().c_str());
			cairo_stroke(cr);
		}
	}

	int xp = (width-ax_out) * (play_t - show_s_sec) / show_t_sec + ax_out;
	if(xp >= ax_out && xp < width){
		cairo_set_source_rgb(cr, 1, 1, 0);
		cairo_set_line_width(cr, 2);
		cairo_move_to(cr, left+xp, top);
		cairo_line_to(cr, left+xp, top+height-ax_out+1);
		cairo_stroke(cr);

		std::ostringstream out;
		out << std::fixed << std::setprecision(3) << play_t << "sec";
		cairo_text_extents_t extents;
		cairo_text_extents(cr, out.str().c_str(), &extents);

		cairo_set_source_rgb(cr, 1, 1, 0);
		cairo_move_to(cr, left+xp-extents.width/2, top+height-ax_out-extents.height);
		cairo_show_text(cr, out.str().c_str());

		cairo_stroke(cr);
	}

	return !notyet;
}

void Converter::DrawWave(cairo_t *cr, const struct drawarea &area, double start_sec, double show_sec, double y_ampmax, double select_fq)
{

	int top = area.top;
	int left = area.left;
	int height = area.height;
	int width = area.width;

	int ax_out = 20;

	int spec_span = spec_call_sec * samplefq;
	int offset = (start_sec - show_sec/2) * samplefq;
	int length = show_sec * samplefq;
	if(length > maxsamples) length = maxsamples;
	if(offset < 0) offset = 0;
	if(offset + length > maxsamples) offset = maxsamples - length;
	start_sec = (double)offset / samplefq;
	show_sec = (double)length / samplefq;

	cairo_set_source_rgb(cr, 0, 0, 0);
	cairo_rectangle(cr, left, top, width, height);
	cairo_fill(cr);

	cairo_set_source_rgb(cr, 1, 1, 1);
	cairo_set_line_width(cr, 1);

	cairo_move_to(cr, left+ax_out-1, top);
	cairo_line_to(cr, left+ax_out-1, top+height-ax_out+1);
	cairo_line_to(cr, left+width, top+height-ax_out+1);

	int wave_y = (height-ax_out)/3;
	int wave_base = top + wave_y / 2;
	int amp_base = top + wave_y / 2 + wave_y;
	int phase_base = top + wave_y / 2 + 2* wave_y;

	cairo_move_to(cr, left+ax_out, wave_base);
	cairo_line_to(cr, left+width, wave_base);
	cairo_move_to(cr, left+ax_out, amp_base);
	cairo_line_to(cr, left+width, amp_base);
	cairo_move_to(cr, left+ax_out, phase_base);
	cairo_line_to(cr, left+width, phase_base);


	const int tic_l = 10;

	double t2 = floor(start_sec * 10) / 10;
	for(int x = left+ax_out; x < left+width; x++){
		double t1 = (double)(x-ax_out-left) / (width-ax_out) * show_sec + start_sec;
		if(t2 < t1){
			t2 = floor(t1 * 10 + 1)/10;
			cairo_move_to(cr, left+x, top+height-ax_out+1);
			cairo_line_to(cr, left+x, top+height-ax_out+tic_l);

			std::ostringstream out;
			out << std::fixed << std::setprecision(1) << t1;
			cairo_text_extents_t extents;
			cairo_text_extents(cr, out.str().c_str(), &extents);

			cairo_move_to(cr, left+x-extents.width/2, top+height-ax_out+tic_l+extents.height);
			cairo_show_text(cr, out.str().c_str());

			while((t2 - t1) / show_sec * (width-ax_out) < extents.width*2){
				t2 = floor(t2 * 10 + 1)/10;
			}
		}
	}
	cairo_stroke(cr);
	
	int f = -1;
	if(select_fq > 0){
		while(++f < fm && omh[f] < select_fq);
	}
	if(f < 0 || f >= fm) return;

	std::vector<int> w_y;
	std::vector<int> a_y;
	std::vector<int> p_y;
	double lastphase = 0;
	if(IsFilterRun){
		std::unique_lock<std::mutex> lock(mtx_fspec, std::try_to_lock);
		if(lock.owns_lock()){
			std::unique_lock<std::mutex> lock2(mtx_spec, std::try_to_lock);
			if(lock2.owns_lock()){
				for(int x = ax_out; x < width; x++){
					double t1 = (double)(x-ax_out) / (width-ax_out) * show_t_sec + show_s_sec;
					int t2 = t1 * samplefq;
					int t3 = t2 % spec_span;
					int slot = t2 / spec_span;

					double specvalue = 0;
					double phasevalue = 0;
					if(!fspec_map[slot].empty() && t3*fm+f < (int)fspec_map[slot].size()){
						specvalue = fspec_map[slot][t3*fm + f];
					}
					else if (!spec_map[slot].empty() && t3*fm+f < (int)spec_map[slot].size()){
						specvalue = spec_map[slot][t3*fm + f];
					}
					if(!aasc_map[slot].empty() && t3*fm+f < (int)aasc_map[slot].size()){
						phasevalue = aasc_map[slot][t3*fm + f];
					}
					
					double value = specvalue * sin(phasevalue);
					double dphase = phasevalue - lastphase;
					while(dphase > 2*M_PI) dphase -= 2*M_PI;
					while(dphase < 0) dphase += 2*M_PI;
					lastphase = phasevalue;

					double svalue = (specvalue > 0)? 20*log(specvalue) + 80: 0;
					if(svalue < -40) svalue = -40;

					w_y.push_back( -(value/0.2) * wave_y / 2 + wave_base);
					a_y.push_back( -(svalue/40) * wave_y / 2 + amp_base);
					p_y.push_back( -(dphase - (2*M_PI*omh[f]/samplefq))/(2*M_PI) * wave_y / 2 + phase_base);
				}
			}
		}
	}
	else{
		std::unique_lock<std::mutex> lock(mtx_spec, std::try_to_lock);
		if(lock.owns_lock()){
			for(int x = ax_out; x < width; x++){
				double t1 = (double)(x-ax_out) / (width-ax_out) * show_t_sec + show_s_sec;
				int t2 = t1 * samplefq;
				int t3 = t2 % spec_span;
				int slot = t2 / spec_span;

				double specvalue = 0;
				double phasevalue = 0;
				if (!spec_map[slot].empty() && t3*fm+f < (int)spec_map[slot].size()){
					specvalue = spec_map[slot][t3*fm + f];
				}
				if(!aasc_map[slot].empty() && t3*fm+f < (int)aasc_map[slot].size()){
					phasevalue = aasc_map[slot][t3*fm + f];
				}

				double value = specvalue * sin(phasevalue);
				double dphase = phasevalue - lastphase;
				while(dphase > 2*M_PI) dphase -= 2*M_PI;
				while(dphase < 0) dphase += 2*M_PI;
				lastphase = phasevalue;
				
				double svalue = (specvalue > 0)? 20*log(specvalue) + 80: 0;
				if(svalue < -40) svalue = -40;

				w_y.push_back( -(value/0.2) * wave_y / 2 + wave_base);
				a_y.push_back( -(svalue/40) * wave_y / 2 + amp_base);
				p_y.push_back( -(dphase - (2*M_PI*omh[f]/samplefq))/(2*M_PI) * wave_y / 2 + phase_base);
			}
		}
	}
	
	cairo_set_source_rgb(cr, 0, 1, 0);
	cairo_set_line_width(cr, 1);

	cairo_move_to(cr, left+ax_out, wave_base);
	for(int i=0; i< w_y.size(); i++){
		int x = i + ax_out + left;
		cairo_line_to(cr, x, w_y[i]);
	}
	cairo_stroke(cr);

	cairo_set_source_rgb(cr, 0.5, 0.5, 0);
	cairo_set_line_width(cr, 1);

	cairo_move_to(cr, left+ax_out, amp_base);
	for(int i=0; i< a_y.size(); i++){
		int x = i + ax_out + left;
		cairo_line_to(cr, x, a_y[i]);
	}
	cairo_stroke(cr);
	
	cairo_set_source_rgb(cr, 0, 0.5, 0.5);
	cairo_set_line_width(cr, 1);

	cairo_move_to(cr, left+ax_out, phase_base);
	for(int i=0; i< p_y.size(); i++){
		int x = i + ax_out + left;
		cairo_line_to(cr, x, p_y[i]);
	}
	cairo_stroke(cr);
}

void Converter::fill_spec()
{
	int offset = show_s_sec * samplefq;
	int len = show_t_sec * samplefq;
	int spec_span = spec_call_sec * samplefq;
	
	for(int i = offset/spec_span-1; i <= (offset + len)/spec_span+3; i++){
		if(i<0) continue;
		if(i > ceil((double)maxsamples/samplefq/spec_call_sec)) continue;
		if(spec_max[i] < 0){
			std::lock_guard<std::mutex> lock1(mtx_analyse);
			wait_analyse.push_back(i);
			wait_analyse.sort();
			wait_analyse.unique();
			cv_analyse.notify_all();
		}
		if(IsFilterRun && fspec_max[i] < 0){
			std::lock_guard<std::mutex> lock1(mtx_filter);
			wait_filter.push_back(i);
			wait_filter.sort();
			wait_filter.unique();
			cv_filter.notify_all();
		}
	}
}

bool Converter::IsLoadReady(double target_t_sec, bool filter)
{
	int slot = floor(target_t_sec / spec_call_sec);
	int maxslot = floor(maxsamples / spec_call_sec / samplefq);
	if(slot < 0)
		slot = 0;
	if(slot > maxslot)
		slot = maxslot;
	if(slot == 0){
		for(int i = 0; i <= MIN(maxslot, 2); i++){
			if(spec_max[i] < 0){
				std::lock_guard<std::mutex> lock1(mtx_analyse);
				wait_analyse.push_back(i);
				wait_analyse.sort();
				wait_analyse.unique();
				cv_analyse.notify_all();
			}
			if(filter && fspec_max[i] < 0){
				std::lock_guard<std::mutex> lock1(mtx_filter);
				wait_filter.push_back(i);
				wait_filter.sort();
				wait_filter.unique();
				cv_filter.notify_all();
			}
		}
	}
	else{
		if(spec_max[slot] < 0){
			std::lock_guard<std::mutex> lock1(mtx_analyse);
			wait_analyse.push_back(slot);
			wait_analyse.sort();
			wait_analyse.unique();
			cv_analyse.notify_all();
		}
		if(filter && fspec_max[slot] < 0){
			std::lock_guard<std::mutex> lock1(mtx_filter);
			wait_filter.push_back(slot);
			wait_filter.sort();
			wait_filter.unique();
			cv_filter.notify_all();
		}
	}
	if(slot == 0){
		bool p = true;
		for(int i = 0; i <= MIN(2, maxslot); i++)
			p &= SpecReady(i);
		if(filter){
			for(int i = 0; i <= MIN(2, maxslot); i++)
				p &= FspecReady(i);
		}
		return p;
	}
	else{
		if(filter)
			return SpecReady(slot) && FspecReady(slot);
		else
			return SpecReady(slot);
	}
}

bool Converter::IsReady(double target_t_sec)
{
	int slot = floor(target_t_sec / spec_call_sec);
	int maxslot = floor(maxsamples / spec_call_sec / samplefq);
	if(slot < 0)
		slot = 0;
	if(slot > maxslot)
		slot = maxslot;
	if(slot == 0){
		bool p = true;
		for(int i = 0; i <= MIN(2, maxslot); i++)
			p &= SpecReady(i);
		if(IsFilterRun){
			for(int i = 0; i <= MIN(2, maxslot); i++)
				p &= FspecReady(i);
		}
		return p;
	}
	else{
		if(IsFilterRun)
			return SpecReady(slot) && FspecReady(slot);
		else
			return SpecReady(slot);
	}
}

double Converter::ScrollToTime(double target_t_sec)
{
	if(target_t_sec < 0) return 0;

	double new_show_t = show_s_sec;
	if(target_t_sec < new_show_t)
		new_show_t = floor(target_t_sec / show_t_sec)*show_t_sec; 
	while(target_t_sec > new_show_t + show_t_sec * 0.75){
		new_show_t += show_t_sec * 0.5; 
	}
	if((new_show_t - show_t_sec) * samplefq > maxsamples)
		new_show_t = (double)maxsamples/samplefq - show_t_sec;
	return new_show_t;
}

const std::vector<float>& Converter::GetSpec(int slot)
{
	std::lock_guard<std::mutex> lock(mtx_spec);
	return spec_map[slot];
}

const std::vector<float>& Converter::GetFspec(int slot)
{
	std::lock_guard<std::mutex> lock(mtx_fspec);
	return fspec_map[slot];
}

const std::vector<float>& Converter::GetAasc(int slot)
{
	std::lock_guard<std::mutex> lock(mtx_spec);
	return aasc_map[slot];
}
