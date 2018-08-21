#pragma once

#include "fmrs.hpp"
#include "converter.hpp"

#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#include <pulse/pulseaudio.h>

class SpecReconstructor
{
	private:
		Converter &conv1;
		std::thread thread;

		bool is_alive;
		pa_stream *stream = NULL;
		bool isfiltered;

		const int samplefq;
		const int maxsamples;
		const int fm;
		
		double autogain = 1.0;

		std::vector<bool> wave_converted;

		std::mutex mtx_wave;
		double slowfc = 1.0;
		std::vector<float> play_sound;
		std::map<int, std::vector<double> > phase_map;
		std::map<int, std::vector<double> > amp_map;
		std::vector<float> lastamplist;

		std::mutex mtx_ramps;
		std::map<int, std::vector<std::vector<std::pair<int, float> > > > ramps_map;

		std::list<int> wait_waveconvert;
		std::mutex mtx_waveconvert;
		std::condition_variable cv_waveconvert;

		void convert_thread();
	public:
		SpecReconstructor(Converter &converter1);
		~SpecReconstructor();
	
		const float *data;

		const std::vector<std::vector<std::pair<int, float> > >& GetRamps(int slot)
		{
			std::lock_guard<std::mutex> lock(mtx_ramps);
			return ramps_map[slot];
		}
		bool WantToPlay(double play_point);
};
	

class Player
{
	friend void stream_write_cb(pa_stream *stream, size_t requested_bytes, void *userdata);

	public:
		enum class Reconstruction {
			None,
			Spec,
			Filtered,
		};
		struct drawarea {
			int left;
			int top;
			int width;
			int height;
		};

	private:
		const int ax_out = 30;
		Converter *conv1;
		SpecReconstructor *reconv1;
		Converter *conv2;
		SpecReconstructor *reconv2;

		bool ch1showing;
		bool ch2showing;

		int samplefq;
		int maxsamples;
		int fm;
		pa_stream *stream = NULL;

		int play_buf_idx = 0;
		bool play_pause = true;
		uint64_t pause_count = 0;
		uint64_t play_count = 0;
		double play_start_point = 0;

		const float *left;
		const float *right;

		bool want_to_play = false;

		double slowfc = 1.0;
		void playsound();

		void DrawSpecAdd(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max, SpecReconstructor &rcont1);
		void print_HzdB(cairo_t *cr, int idx, double Hz, double value, const struct drawarea &area, double y_range, double y_min);
	public:
		Player(Converter &converter1);
		Player(Converter &converter1, Converter &converter2);
		Player(Converter &converter1, Reconstruction reconst1);
		Player(Converter &converter1, Reconstruction reconst1, Converter &converter2, Reconstruction reconst2);
		~Player();

		Tonality tonality = Tonality::None;
		double A440 = 440;

		void ShowChannel(bool conv1, bool conv2);
		double GetNowPlayTime();
		bool WantToPlay(double play_point);
		void Play(double play_point);
		void Pause();
		void DrawWave(cairo_t *cr, const struct drawarea &area, double start_sec, double show_sec, double y_ampmax, bool isleft);
		void DrawSpec(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max);
};
