#pragma once

#include "wave.hpp"

#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

#include <functional>
#include <vector>

class Player;

class Converter
{
	friend class SpecReconstructor;
	friend class Player;

	public:
		struct drawarea {
			int left;
			int top;
			int width;
			int height;
		};
	private:
		std::thread thread;
		std::thread thread2;
		
		std::list<int> wait_analyse;
		std::mutex mtx_analyse;
		std::condition_variable cv_analyse;

		std::list<int> wait_filter;
		std::mutex mtx_filter;
		std::condition_variable cv_filter;


		GdkPixbuf *pixbuf;

		void spec_thread();
		void filter_thread();
		void fill_spec();
		void print_HzdB(cairo_t *cr, int idx, double Hz, double value, const struct drawarea &area, double y_range, double y_min);
		std::vector<float> findmask(double Hz, double value);
	protected:
		int spec_slotcount = 0;
		std::mutex mtx_spec;
		std::map<int, std::vector<float> > spec_map;
		std::map<int, std::vector<float> > aasc_map;
		int fspec_slotcount = 0;
		std::mutex mtx_fspec;
		std::map<int, std::vector<float> > fspec_map;

		int converting_slot = 0;
		double show_t_sec;
		double show_s_sec;

		bool is_alive;
		int samplefq;
		int maxsamples;
		int fm;
		std::vector<float> omh;
		std::vector<float> spec_max;
		std::vector<float> fspec_max;

		float *data;
		float *datap;

		virtual void convert(std::vector<float> &spec, std::vector<float> &aasc, int slot, int offset, int reqlen) = 0;
		virtual void filter(const std::vector<float> &spec, std::vector<float> &fspec) = 0;
	public:
		const int ax_out = 50;
		double A440 = 440.0;
		Tonality tonality = Tonality::None;
		std::function<void(int slot)> convert_done;
		std::function<void(int slot)> filter_done;
		bool IsFilterRun = false;

		Converter();
		virtual ~Converter();

		bool LoadData(double show_t_sec, double show_s_sec);
		bool DrawImage(cairo_t *cr, const struct drawarea &area, double show_t_sec, double show_s_sec, double play_t, double select_fq, double min_freq = -1, double max_freq = -1); 
		bool DrawSpec(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max, bool main = true); 
		bool DrawSpecAdd(cairo_t *cr, const struct drawarea &area, double target_sec, double y_min, double y_max, bool main = true); 
		void DrawWave(cairo_t *cr, const struct drawarea &area, double start_sec, double show_sec, double y_ampmax, double select_fq);
		bool IsReady(double target_t_sec);
		bool IsLoadReady(double target_t_sec, bool filter);
		double ScrollToTime(double target_t_sec);

		const std::vector<float>& GetSpec(int slot);
		const std::vector<float>& GetFspec(int slot);
		const std::vector<float>& GetAasc(int slot);
		int GetSamplefq() const { return samplefq; }
		int GetMaxsamples() const { return maxsamples; }
		int GetFm() const { return fm; }
		const std::vector<float>& GetOmh() const { return omh; }
		bool SpecReady(int slot) { return (int)spec_max.size() > slot && spec_max[slot] >= 0; }
		bool FspecReady(int slot) { return (int)fspec_max.size() > slot && fspec_max[slot] >= 0; }
};
