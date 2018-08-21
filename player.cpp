#include "player.hpp"

#include <future>
#include <thread>

void context_state_cb(pa_context* context, void* mainloop) 
{
	pa_threaded_mainloop_signal(static_cast<pa_threaded_mainloop*>(mainloop), 0);
}

void stream_state_cb(pa_stream *s, void *mainloop) 
{
	pa_threaded_mainloop_signal(static_cast<pa_threaded_mainloop*>(mainloop), 0);
}

void stream_write_cb(pa_stream *stream, size_t requested_bytes, void *userdata) 
{
	Player *player = (Player *)userdata;

	size_t bytes_remaining = requested_bytes;
	while (bytes_remaining > 0) {
		uint16_t *buffer = NULL;
		size_t bytes_to_fill = player->samplefq*2*sizeof(int16_t);

		if (bytes_to_fill > bytes_remaining) bytes_to_fill = bytes_remaining;

		if(player->want_to_play){
			double t = player->GetNowPlayTime() + 2.0;
			if(t >= 0 && t < (double)player->maxsamples/player->samplefq)
				player->play_pause = !player->WantToPlay(t);
			else
				player->play_pause = !player->WantToPlay((double)player->play_buf_idx/player->samplefq);
		}

		pa_stream_begin_write(stream, (void**) &buffer, &bytes_to_fill);

		for (size_t i = 0; i < bytes_to_fill/sizeof(int16_t); i += 2) {
			player->play_count++;
			if(player->play_pause){
				player->pause_count++;
				buffer[i] = buffer[i+1] = 0;
			}
			else{
				float value_left = player->left[player->play_buf_idx];
				float value_right = player->right[player->play_buf_idx];
				
				if(value_left > 1.0) value_left = 1.0;
				else if(value_left < -1.0) value_left = -1.0;
				if(value_right > 1.0) value_right = 1.0;
				else if(value_right < -1.0) value_right = -1.0;
				
				buffer[i] = value_left * 0x7f00;
				buffer[i+1] = value_right * 0x7f00;
				
				if(++player->play_buf_idx >= player->maxsamples*player->slowfc){
					player->play_buf_idx = 0;
					player->pause_count = player->play_count;
					player->play_start_point = 0;
				}
			}
		}

		pa_stream_write(stream, buffer, bytes_to_fill, NULL, 0LL, PA_SEEK_RELATIVE);

		bytes_remaining -= bytes_to_fill;
	}
}

void stream_success_cb(pa_stream *stream, int success, void *userdata) 
{
	return;
}


void Player::playsound()
{
	pa_threaded_mainloop *mainloop;
	pa_mainloop_api *mainloop_api;
	pa_context *context;

	// Get a mainloop and its context
	mainloop = pa_threaded_mainloop_new();
	assert(mainloop);
	mainloop_api = pa_threaded_mainloop_get_api(mainloop);
	context = pa_context_new(mainloop_api, "pcm-playback");
	assert(context);

	// Set a callback so we can wait for the context to be ready
	pa_context_set_state_callback(context, &context_state_cb, mainloop);

	// Lock the mainloop so that it does not run and crash before the context is ready
	pa_threaded_mainloop_lock(mainloop);

	// Start the mainloop
	assert(pa_threaded_mainloop_start(mainloop) == 0);
	assert(pa_context_connect(context, NULL, PA_CONTEXT_NOAUTOSPAWN, NULL) == 0);

	// Wait for the context to be ready
	for(;;) {
		pa_context_state_t context_state = pa_context_get_state(context);
		assert(PA_CONTEXT_IS_GOOD(context_state));
		if (context_state == PA_CONTEXT_READY) break;
		pa_threaded_mainloop_wait(mainloop);
	}

	// Create a playback stream
	pa_sample_spec sample_specifications;
	sample_specifications.format = PA_SAMPLE_S16LE;
	sample_specifications.rate = samplefq;
	sample_specifications.channels = 2;

	pa_channel_map map;
	pa_channel_map_init_stereo(&map);

	stream = pa_stream_new(context, "Playback", &sample_specifications, &map);
	pa_stream_set_state_callback(stream, stream_state_cb, mainloop);
	pa_stream_set_write_callback(stream, stream_write_cb, this);

	// recommended settings, i.e. server uses sensible values
	pa_buffer_attr buffer_attr;
	buffer_attr.maxlength = (uint32_t) -1;
	buffer_attr.tlength = (uint32_t) -1;
	buffer_attr.prebuf = (uint32_t) -1;
	buffer_attr.minreq = (uint32_t) -1;

	// Settings copied as per the chromium browser source
	pa_stream_flags_t stream_flags;
	stream_flags = static_cast<pa_stream_flags_t>(PA_STREAM_START_CORKED | PA_STREAM_INTERPOLATE_TIMING |
			PA_STREAM_NOT_MONOTONIC | PA_STREAM_AUTO_TIMING_UPDATE |
			PA_STREAM_ADJUST_LATENCY);

	// Connect stream to the default audio output sink
	assert(pa_stream_connect_playback(stream, NULL, &buffer_attr, stream_flags, NULL, NULL) == 0);

	// Wait for the stream to be ready
	for(;;) {
		pa_stream_state_t stream_state = pa_stream_get_state(stream);
		assert(PA_STREAM_IS_GOOD(stream_state));
		if (stream_state == PA_STREAM_READY) break;
		pa_threaded_mainloop_wait(mainloop);
	}

	pa_threaded_mainloop_unlock(mainloop);

	// Uncork the stream so it will start playing
	std::async(std::launch::async, [this, mainloop]{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			pa_stream_cork(stream, 0, stream_success_cb, mainloop);
			});
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

Player::Player(Converter &converter1)
	: conv1(&converter1), reconv1(NULL), conv2(NULL), reconv2(NULL)
{
	samplefq = conv1->samplefq;
	maxsamples = conv1->maxsamples;
	fm = conv1->fm;
	left = conv1->data;
	right = conv1->data;	
	
	playsound();
}

Player::Player(Converter &converter1, Reconstruction reconst1)
	: conv1(&converter1), reconv1((reconst1 != Reconstruction::None)? new SpecReconstructor(*conv1): NULL), 
	conv2(NULL), reconv2(NULL)
{
	samplefq = conv1->samplefq;
	maxsamples = conv1->maxsamples;
	fm = conv1->fm;
	left = (reconst1 != Reconstruction::None)? reconv1->data: conv1->data;
	right = (reconst1 != Reconstruction::None)? reconv1->data: conv1->data;	
	
	playsound();
}

Player::Player(Converter &converter1, Converter &converter2)
	: conv1(&converter1), reconv1(NULL), conv2(&converter2), reconv2(NULL)
{
	samplefq = conv1->samplefq;
	maxsamples = conv1->maxsamples;
	fm = conv1->fm;
	left = conv1->data;
	right = conv2->data;	
	
	playsound();
}

Player::Player(Converter &converter1, Reconstruction reconst1, Converter &converter2, Reconstruction reconst2)
	: conv1(&converter1), reconv1((reconst1 != Reconstruction::None)? new SpecReconstructor(*conv1): NULL), 
	conv2(&converter2), reconv2((reconst2 != Reconstruction::None)? new SpecReconstructor(*conv2): NULL)
{
	samplefq = conv1->samplefq;
	maxsamples = conv1->maxsamples;
	fm = conv1->fm;
	left = (reconst1 != Reconstruction::None)? reconv1->data: conv1->data;
	right = (reconst2 != Reconstruction::None)? reconv2->data: conv2->data;	
	
	playsound();
}

Player::~Player()
{
	printf("Player shutdown...\n");
	pa_stream_cork(stream, 1, stream_success_cb, NULL);

	delete(reconv1);
	delete(reconv2);
}

void Player::ShowChannel(bool conv1, bool conv2)
{
	ch1showing = conv1;
	ch2showing = conv2;
}

double Player::GetNowPlayTime()
{
	if(stream && want_to_play){
		pa_usec_t t_usec;
		pa_stream_get_time(stream, &t_usec);
		double play_now_t = t_usec / 1000000.0 - (double)pause_count / samplefq + play_start_point;
		play_now_t /= slowfc;
		return play_now_t;
	}
	return -100;
}

void Player::Play(double play_point)
{
	want_to_play = true;
	if(reconv1)
		reconv1->WantToPlay(play_point);
	if(reconv2)
		reconv2->WantToPlay(play_point);
	play_buf_idx = play_point * samplefq * slowfc;
	play_start_point = play_point;
}

bool Player::WantToPlay(double play_point)
{
	if(play_point < 0) play_point = 0;
	double maxplay = (double)maxsamples/samplefq;
	if(play_point > maxplay) play_point = 0;
	want_to_play = true;
	bool ret = true;
	if(conv1 && ch1showing)
		ret &= conv1->IsReady(play_point);
	if(conv2 && ch2showing)
		ret &= conv2->IsReady(play_point);
	if(reconv1)
		ret &= reconv1->WantToPlay(play_point);
	if(reconv2)
		ret &= reconv2->WantToPlay(play_point);
	return ret;
}

void Player::Pause()
{
	want_to_play = false;
	play_pause = true;
	pause_count = play_count;
}


void Player::DrawWave(cairo_t *cr, const struct drawarea &area, double start_sec, double show_sec, double y_ampmax, bool isleft)
{

	int top = area.top;
	int left = area.left;
	int height = area.height;
	int width = area.width;

	int ax_out = 20;

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

	const int tic_l = 10;
	for(double v = -10; v <= 10; v++){
		int y = height - ax_out + top - (v * 0.1 + 1) / 2 * (height - ax_out);
		cairo_move_to(cr, left+ax_out-tic_l, top+y);
		cairo_line_to(cr, left+ax_out-1, top+y);

		std::stringstream out;
		out << std::fixed << std::setprecision(1) << (v * 0.1);
		std::string tag = out.str();

		cairo_text_extents_t extents;
		cairo_text_extents(cr, tag.c_str(), &extents);

		cairo_move_to(cr, left, y - extents.y_bearing/2);
		cairo_show_text(cr, tag.c_str());
	}

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


	cairo_set_source_rgb(cr, 0, 1, 0);
	cairo_set_line_width(cr, 1);

	cairo_move_to(cr, left+ax_out, top+height-ax_out-(height-ax_out)/2);
	for(int t = offset; t < offset+length; t++){
		double x = (double)(t - offset) / length * (width-ax_out) + left + ax_out;
		double value = (isleft)? this->left[t]: this->right[t];
		double y = -(value/y_ampmax + 1) / 2 * (height - ax_out) + height - ax_out + top;

		cairo_line_to(cr, x, y);
	}
	cairo_stroke(cr);

}

void Player::print_HzdB(cairo_t *cr, int idx, double Hz, double value, const struct drawarea &area, double y_range, double y_min)
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


void Player::DrawSpecAdd(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max, SpecReconstructor &rcont1)
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
	int slot = offset/spec_span;
	int index = offset - slot*spec_span;
	
	if(offset >= 0 && offset < maxsamples && rcont1.GetRamps(slot).size() > index){
		auto rspec = rcont1.GetRamps(slot)[index];
		for(const auto &ra: rspec){
			int f = ra.first;
			int x = (double)f/fm * (width-ax_out) + ax_out + left;
			float value = ra.second;
			if(value > 0)
				value = log10(value);
			else
				value = y_min;
			if(value < y_min)
				value = y_min;
			int y = height - ax_out + top - (value - y_min) / y_range * (height - ax_out);
			cairo_set_source_rgb(cr, 0, 1, 0);
			cairo_set_line_width(cr, 1);
			cairo_move_to(cr, x, height - ax_out + top);
			cairo_line_to(cr, x, y);
			cairo_stroke(cr);
			print_HzdB(cr, ra.first, conv1->omh[ra.first], ra.second, area, y_range, y_min);
		}
	}
}

void Player::DrawSpec(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max)
{
	int top = area.top;
	int left = area.left;
	int height = area.height;
	int width = area.width;

	cairo_set_source_rgb(cr, 0, 0, 0);
	cairo_rectangle(cr, left, top, width, height);
	cairo_fill(cr);

	if(reconv1)
		DrawSpecAdd(cr, area, target_sec, y_min, y_max, *reconv1);

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
		while(!xtics.empty() && conv1->omh[f1] >= xtics.front()){
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
		while(!pitch.empty() && conv1->omh[f1] >= pitch.front().first){
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
}
/////////////////////////////////////////////////////////////////////////////


SpecReconstructor::SpecReconstructor(Converter &converter1)
	: conv1(converter1), samplefq(conv1.samplefq),
	maxsamples(conv1.maxsamples), fm(conv1.fm)
{
	is_alive = true;

	wave_converted.resize(ceil((double)maxsamples/samplefq/spec_call_sec), false);
	play_sound.resize(maxsamples*slowfc);
	data = play_sound.data();
	
	thread = std::thread(&SpecReconstructor::convert_thread, this);
}

SpecReconstructor::~SpecReconstructor()
{
	printf("SpecReconstructor shutdown...\n");
	is_alive = false;
	{
		std::lock_guard<std::mutex> lock(mtx_waveconvert);
		wait_waveconvert.push_back(0);
		cv_waveconvert.notify_all();
	}
	thread.join();
}

void SpecReconstructor::convert_thread()
{
	conv1.filter_done = [this](int slot){
		std::lock_guard<std::mutex> lock(mtx_waveconvert);
		wait_waveconvert.push_back(slot);
		cv_waveconvert.notify_all();
	};

	while(is_alive){
		int slot_idx;
		{
			std::unique_lock<std::mutex> lock(mtx_waveconvert);
			cv_waveconvert.wait(lock, [this]{ return !wait_waveconvert.empty(); });
			slot_idx = wait_waveconvert.front();
			wait_waveconvert.pop_front();
		}
		if(!is_alive) return;
	
		int spec_span = spec_call_sec * samplefq;
		int offset = slot_idx * spec_span;
		int reqlen = (offset + spec_call_sec*samplefq < maxsamples)? spec_call_sec*samplefq: maxsamples - offset;
	
		if(wave_converted[slot_idx]) continue;
		
		if(!conv1.IsLoadReady(slot_idx * spec_call_sec, true)){
			std::async(std::launch::async, [this, slot_idx]{
					std::this_thread::sleep_for(std::chrono::seconds(1));
					std::lock_guard<std::mutex> lock(mtx_waveconvert);
					wait_waveconvert.push_back(slot_idx);
					cv_waveconvert.notify_all();
				});
			continue;
		}

		while(!conv1.FspecReady(slot_idx)){
			if(!is_alive) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
		}
		while(!conv1.SpecReady(slot_idx)){
			if(!is_alive) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
		}
		if(slot_idx > 0){
			while(!conv1.SpecReady(slot_idx-1)){
				if(!is_alive) break;
				std::this_thread::sleep_for(std::chrono::milliseconds(250));
			}
		}

		if(!is_alive) break;
		std::vector<double> *prev_phase = NULL;
		std::vector<double> *target_phase;
		std::vector<double> *prev_amp = NULL;
		std::vector<double> *target_amp;
		double slowfct;
		{
			std::lock_guard<std::mutex> lock(mtx_wave);
			if(slot_idx > 0){
				prev_phase = &phase_map[slot_idx-1];
				prev_amp = &amp_map[slot_idx-1];
			}
			target_phase = &phase_map[slot_idx];
			target_amp = &amp_map[slot_idx];
			slowfct = slowfc;
		}

		std::vector<float> *target_spec;
		std::vector<float> *target_fspec;
		{
			std::scoped_lock(conv1.mtx_spec, conv1.mtx_fspec);
			target_fspec = &conv1.fspec_map[slot_idx];
			conv1.converting_slot = slot_idx;
		}
		const int movave = 10;
		if(lastamplist.size() != fm*movave){
			lastamplist.resize(fm*movave);
		}
		{
			std::lock_guard<std::mutex> lock(conv1.mtx_spec);
			if(slot_idx > 0){
				int len = conv1.spec_map[slot_idx-1].size() / fm;
				int st = len - movave;
				if(st < 0){
					st = 0;
				}
				else{
					len = movave;
				}
				memcpy(&lastamplist[(movave-len)*fm], &conv1.spec_map[slot_idx-1][st*fm], sizeof(float)*len*fm);
			}
			target_spec = &conv1.spec_map[slot_idx];
			conv1.converting_slot = slot_idx;
		}
		std::vector<float> lastamp(fm);
		for(int u = 0; u < fm; u++){
			for(int i = 0; i < movave; i++){
				lastamp[u] += lastamplist[i*fm+u];
			}
		}
		int lastidx = 0;
		std::vector<float> tmp_wave(reqlen*slowfct);
		if(prev_phase && (int)prev_phase->size() == fm)
			*target_phase = *prev_phase;
		else
			target_phase->resize(fm);
		if(prev_amp && (int)prev_amp->size() == fm)
			*target_amp = *prev_amp;
		else
			target_amp->resize(fm);

		int gain_graw = 0;
		size_t t_out = 0;
		const double cutoff = -50;
		std::vector<std::vector<std::pair<int, float> > > tmp_ramp(target_spec->size()/fm);
		std::vector<std::pair<int, float> > prevamp;
		{
			std::vector<int> ampidx(fm);
			double *ampp = target_amp->data();
			for(int u = 0; u < fm; u++){
				ampidx[u] = u;
			}
			std::sort(ampidx.begin(), ampidx.end(), [ampp](const int& x, const int &y){ return ampp[x] > ampp[y]; });
			std::vector<std::pair<int, float> > amp;
			for(const auto &ai: ampidx){
				if(ampp[ai] <= 0 || 20*log10(ampp[ai]) < cutoff) continue;
				amp.push_back(std::make_pair(ai, ampp[ai]));
			}
			prevamp = amp;
		}
		const int search_area = 5;
		for(size_t t = 0; t < target_spec->size()/fm; t++){
			if(!is_alive) return;
			std::vector<float> avespec(fm);
			float *ampp = &(*target_spec)[t*fm];
			float *fampp = &(*target_fspec)[t*fm];
			for(int u = 0; u < fm; u++){
				lastamp[u] -= lastamplist[lastidx*fm+u];
				lastamplist[lastidx*fm+u] = ampp[u];
				lastamp[u] += ampp[u];
				avespec[u] = lastamp[u] / movave;
			}
			if(++lastidx >= movave) lastidx = 0;

			std::vector<int> ampidx(fm);
			std::vector<bool> activate(fm, false);
			for(int u = 0; u < fm; u++){
				ampidx[u] = u;
			}
			std::sort(ampidx.begin(), ampidx.end(), [fampp](const int& x, const int &y){ return fampp[x] > fampp[y]; });
			std::vector<std::pair<int, float> > amp;
			for(const auto &ai: ampidx){
				if(fampp[ai] <= 0){
					break;
				}
				if(activate[ai]) continue;
				std::vector<int> tampidx;
				for(int i = MAX(0, ai-search_area); i < MIN(fm, ai+search_area+1); i++){
					tampidx.push_back(i);
				}
				std::sort(tampidx.begin(), tampidx.end(), [avespec](const int& x, const int &y){ return avespec[x] > avespec[y]; });
				int peak_idx = 0;
				int samecount = 0;
				for(auto const &i: tampidx){
					if(avespec[tampidx[0]] == avespec[i]){
						peak_idx += i;
						samecount++;
					}
					else if(activate[i]){
						peak_idx = i;
						samecount = 1;
						break;
					}
					else{
						break;
					}
				}
				peak_idx /= samecount;
				
				if(!activate[peak_idx]){
					amp.push_back(std::make_pair(peak_idx, avespec[peak_idx]));
					activate[peak_idx] = true;
				}
			}
			if(false){
				for(const auto &pa: prevamp){
					if(activate[pa.first]) continue;
					if(avespec[pa.first] <= 0 || 20*log10(avespec[pa.first]) < cutoff) continue;
					std::vector<int> tampidx;
					for(int i = MAX(0, pa.first-search_area); i < MIN(fm, pa.first+search_area+1); i++){
						tampidx.push_back(i);
					}
					std::sort(tampidx.begin(), tampidx.end(), [avespec](const int& x, const int &y){ return avespec[x] > avespec[y]; });
					int peak_idx = 0;
					int samecount = 0;
					for(auto const &i: tampidx){
						if(avespec[tampidx[0]] == avespec[i]){
							peak_idx += i;
							samecount++;
						}
						else if(activate[i]){
							peak_idx = i;
							samecount = 1;
							break;
						}
						else{
							break;
						}
					}
					peak_idx /= samecount;

					if(!activate[peak_idx]){
						amp.push_back(std::make_pair(peak_idx, avespec[peak_idx]));
						activate[peak_idx] = true;
					}
				}
			}
			if(!prevamp.empty()){
				for(const auto &a: amp){
					int idx = a.first;
					double amp_i = a.second;
					std::sort(prevamp.begin(), prevamp.end(),
							[amp_i](const std::pair<int, float> &x, const std::pair<int, float> &y) { return fabs(x.second - amp_i) < fabs(y.second - amp_i); });
					std::sort(prevamp.begin(), prevamp.end(),
							[idx](const std::pair<int, float> &x, const std::pair<int, float> &y) { return abs(x.first - idx) < abs(y.first - idx); });
					(*target_phase)[idx] = (*target_phase)[prevamp[0].first];
				}
			}
			prevamp = amp;
			tmp_ramp[t] = amp;

			while(t_out <= t*slowfct){
				double wav = 0;
				memset(target_amp->data(), 0, sizeof(double)*fm);
				for(const auto &a: amp){
					int u = a.first;
					double amp1 = a.second;
					(*target_phase)[u] += conv1.omh[u] / samplefq * 2 * M_PI;
					wav += amp1 * sin((*target_phase)[u]);
					(*target_amp)[u] = amp1;
				}
				tmp_wave[t_out] = autogain * wav;
				if(fabs(tmp_wave[t_out]) < 0.5){
					gain_graw++;
				}
				else{
					gain_graw = 0;
				}
				if(fabs(tmp_wave[t_out]) > 0.75){
					autogain /= 1.05;
					tmp_wave[t_out] *= autogain;
				}
				if(gain_graw > samplefq){
					autogain *= 1.05;
					gain_graw = 0;
				}
				t_out++;
			}
		}
		{
			std::lock_guard<std::mutex> lock(mtx_ramps);
			ramps_map[slot_idx] = tmp_ramp;
		}

		if(!is_alive) return;
		{
			std::lock_guard<std::mutex> lock(mtx_wave);
			if(slowfct == slowfc){
				memcpy(&play_sound[offset*slowfc], tmp_wave.data(), sizeof(float)*tmp_wave.size());
				wave_converted[slot_idx] = true;
			}
		}
		conv1.converting_slot = 0;
	}
}

bool SpecReconstructor::WantToPlay(double play_point)
{
	int slot = floor(play_point / spec_call_sec);
	int maxslot = floor((double)maxsamples / samplefq / spec_call_sec);
	if(slot < 0) slot = 0;
	if(slot > maxslot) slot = maxslot;
	if(!wave_converted[slot]){
		std::lock_guard<std::mutex> lock1(mtx_waveconvert);
		wait_waveconvert.push_back(slot);
		wait_waveconvert.sort();
		wait_waveconvert.unique();
		cv_waveconvert.notify_all();
	}
	if(slot+1 <= maxslot && !wave_converted[slot+1]){
		std::lock_guard<std::mutex> lock1(mtx_waveconvert);
		wait_waveconvert.push_back(slot+1);
		wait_waveconvert.sort();
		wait_waveconvert.unique();
		cv_waveconvert.notify_all();
	}
	if(wave_converted[slot]) return true;
	return false;
}

