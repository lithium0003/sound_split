#include <stdio.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <string>
#include <functional>
#include <regex>
#include <sstream>
#include <getopt.h>
#include <iostream>

#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gdk/gdkkeysyms.h>

#include <pulse/pulseaudio.h>

#include "fmrs.hpp"
#include "spec_convert.hpp"
#include "filterspec_convert.hpp"
#include "player.hpp"

static Converter *converter1 = NULL;
static Converter *converter2 = NULL;
static Player *player = NULL;
static GtkWidget *main_drawarea = NULL;
static GtkWidget *spec_drawarea = NULL;
static GtkWidget *spec_window = NULL;
static GtkWidget *wave_drawarea = NULL;
static GtkWidget *wave_window = NULL;
static GtkWidget *test_drawarea = NULL;
static GtkWidget *test_window = NULL;
static GtkWidget *filter_window = NULL;

static double show_s_sec = 0;
static double show_t_sec = 5.0;
static double select_sec = -1;
static bool play = false;

static double select_Hz = -1;

static bool ch1show = true;
static bool ch2show = true;
static bool filtshow = false;
static double A440 = 440.0;
static Tonality key = Tonality::None;

static double freq_min = -1;
static double freq_max = -1;

static double spec_y_min = -70.0;
static double spec_y_max = 0.0;

static double wave_x_range = 5.0;
static double wave_y_max = 1.0;

static gint timer;

int maxsamples;
int samplefq;


static void destroy(GtkWidget *window, gpointer data)
{
        gtk_main_quit();
}

static gboolean delete_event(GtkWidget *window, GdkEvent *event, gpointer data)
{
	g_source_remove(timer);
	return FALSE;
}

static gboolean timer_callback(gpointer data)
{
	if(!data) return TRUE;

	if(play){
		GtkAdjustment *adj1 = GTK_ADJUSTMENT(data);

		double play_now_t = player->GetNowPlayTime();
		if(play_now_t > -100){
			if(play_now_t < 0){
				play_now_t = 0;
			}
			gtk_adjustment_set_value(adj1, converter1->ScrollToTime(play_now_t));
			select_sec = play_now_t;
		}
		player->ShowChannel(ch1show, ch2show);
	}
	if(spec_drawarea)
		gtk_widget_queue_draw(spec_drawarea);
	if(wave_drawarea)
		gtk_widget_queue_draw(wave_drawarea);
	if(test_drawarea)
		gtk_widget_queue_draw(test_drawarea);
	gtk_widget_queue_draw(main_drawarea);
	return TRUE;
}

static gboolean draw_callback(GtkWidget *area, cairo_t *cr, gpointer data)
{
	gint width, height;

	width = gtk_widget_get_allocated_width(area);
	height = gtk_widget_get_allocated_height(area);
	
	if(converter1 && converter2 && ch1show && ch2show){
		double t = player->GetNowPlayTime();
		if(t < 0) t = select_sec;
		Converter::drawarea d = {0, 0, width, height/2};
		converter1->IsFilterRun = filtshow;
		converter1->A440 = A440;
		converter1->tonality = key;
		converter1->DrawImage(cr, d, show_t_sec, show_s_sec, t, select_Hz, freq_min, freq_max);
		d.top += height/2;
		converter2->IsFilterRun = filtshow;
		converter2->A440 = A440;
		converter2->tonality = key;
		converter2->DrawImage(cr, d, show_t_sec, show_s_sec, t, select_Hz, freq_min, freq_max);
	}
	else{
		Converter *c = (converter1 && ch1show)? converter1: (converter2 && ch2show)? converter2: NULL;
		if(c){
			double t = player->GetNowPlayTime();
			if(t < 0) t = select_sec;
			Converter::drawarea d = {0, 0, width, height};
			c->IsFilterRun = filtshow;
			c->A440 = A440;
			c->tonality = key;
			c->DrawImage(cr, d, show_t_sec, show_s_sec, t, select_Hz, freq_min, freq_max);
		}
	}
	return FALSE;
}

static void value_changed_callback(GtkAdjustment *adjustment, gpointer data)
{
	show_s_sec = gtk_adjustment_get_value(adjustment);
	gtk_widget_queue_draw(main_drawarea);
}

static void button_play_clicked(GtkButton *button, gpointer data)
{
	if(!play){
		player->Play((select_sec < 0)? show_s_sec: select_sec);
	}
	play = true;
}

static void button_stop_clicked(GtkButton *button, gpointer data)
{
	if(play)
		player->Pause();
	play = false;
}

static void toggle_button_toggled(GtkToggleButton *source, gpointer data)
{
	*(bool *)data = gtk_toggle_button_get_active(source);
	gtk_widget_queue_draw(main_drawarea);
}

static void entry_activate(GtkEntry *entry, gpointer data)
{
	std::stringstream(gtk_entry_get_text(entry)) >> *(double *)data;
}

static void combo_changed(GtkComboBox *combo, gpointer data)
{
	key = static_cast<Tonality>(gtk_combo_box_get_active(combo));
}

static void spec_destroy(GtkWidget *window, gpointer data)
{
	spec_window = NULL;
	spec_drawarea = NULL;
}

static gboolean spec_draw_callback(GtkWidget *area, cairo_t *cr, gpointer data)
{
	gint width, height;

	width = gtk_widget_get_allocated_width(area);
	height = gtk_widget_get_allocated_height(area);

	if(converter1 && converter2 && ch1show && ch2show){
		Converter::drawarea d = {0, 0, width, height};
		converter2->DrawSpec(cr, d, select_sec, spec_y_min, spec_y_max, false);
		converter1->DrawSpecAdd(cr, d, select_sec, spec_y_min, spec_y_max, true);
	}
	else{
		Converter *c = (converter1 && ch1show)? converter1: (converter2 && ch2show)? converter2: NULL;
		if(c){
			Converter::drawarea d = {0, 0, width, height};
			c->DrawSpec(cr, d, select_sec, spec_y_min, spec_y_max);
		}
	}
	return FALSE;
}


static void button_spectro_clicked(GtkButton *button, gpointer data)
{
	if(!spec_window){
		GtkWidget *vbox, *hbox;
		GtkWidget *label, *entry;

		spec_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

		gtk_window_set_title(GTK_WINDOW(spec_window), "sound_analyse");
		gtk_widget_set_size_request(spec_window, 1200, 800);
		g_signal_connect(spec_window, "destroy", G_CALLBACK(spec_destroy), NULL);

		vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
		gtk_container_add(GTK_CONTAINER(spec_window), vbox);

		spec_drawarea = gtk_drawing_area_new();
		g_signal_connect(spec_drawarea, "draw", G_CALLBACK(spec_draw_callback), NULL);
		gtk_box_pack_start(GTK_BOX(vbox), spec_drawarea, TRUE, TRUE, 0);

		hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
		gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, TRUE, 0);


		label = gtk_label_new("y_min(dB)");
		gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

		entry = gtk_entry_new();
		gtk_entry_set_text(GTK_ENTRY(entry), "-70.0");
		g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &spec_y_min);
		gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);

		label = gtk_label_new("y_max(dB)");
		gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

		entry = gtk_entry_new();
		gtk_entry_set_text(GTK_ENTRY(entry), "0.0");
		g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &spec_y_max);
		gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);
	
		gtk_widget_show_all(spec_window);
	}
}


static void wave_destroy(GtkWidget *window, gpointer data)
{
	wave_window = NULL;
	wave_drawarea = NULL;
}

static gboolean wave_draw_callback(GtkWidget *area, cairo_t *cr, gpointer data)
{
	gint width, height;

	width = gtk_widget_get_allocated_width(area);
	height = gtk_widget_get_allocated_height(area);

	double t = player->GetNowPlayTime();
	if(t < 0) t = select_sec;
	if(t < 0) t = show_s_sec;
	if(converter1 && converter2 && ch1show && ch2show){
		Player::drawarea d = {0, 0, width, height/2};
		player->DrawWave(cr, d, t, wave_x_range, wave_y_max, true);
		d.top += height/2;
		player->DrawWave(cr, d, t, wave_x_range, wave_y_max, false);
	}
	else{
		Player::drawarea d = {0, 0, width, height};
		player->DrawWave(cr, d, t, wave_x_range, wave_y_max, true);
	}
	return FALSE;
}


static void button_wave_clicked(GtkButton *button, gpointer data)
{
	if(!wave_window){
		GtkWidget *vbox, *hbox;
		GtkWidget *label, *entry;

		wave_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

		gtk_window_set_title(GTK_WINDOW(wave_window), "sound_analyse");
		gtk_widget_set_size_request(wave_window, 1200, 800);
		g_signal_connect(wave_window, "destroy", G_CALLBACK(wave_destroy), NULL);

		vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
		gtk_container_add(GTK_CONTAINER(wave_window), vbox);

		wave_drawarea = gtk_drawing_area_new();
		g_signal_connect(wave_drawarea, "draw", G_CALLBACK(wave_draw_callback), NULL);
		gtk_box_pack_start(GTK_BOX(vbox), wave_drawarea, TRUE, TRUE, 0);

		hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
		gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, TRUE, 0);


		label = gtk_label_new("x range(sec)");
		gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

		entry = gtk_entry_new();
		gtk_entry_set_text(GTK_ENTRY(entry), "5.0");
		g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &wave_x_range);
		gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);

		label = gtk_label_new("y range");
		gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

		entry = gtk_entry_new();
		gtk_entry_set_text(GTK_ENTRY(entry), "1.0");
		g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &wave_y_max);
		gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);
	
		gtk_widget_show_all(wave_window);
	}
}


static void test_destroy(GtkWidget *window, gpointer data)
{
	test_window = NULL;
	test_drawarea = NULL;
}

static gboolean test_draw_callback(GtkWidget *area, cairo_t *cr, gpointer data)
{
	gint width, height;

	width = gtk_widget_get_allocated_width(area);
	height = gtk_widget_get_allocated_height(area);

	Converter::drawarea d = {0, 0, width, height};
	Converter *c = NULL;
	if(ch1show && converter1){
		c = converter1;
	}
	else if(ch2show && converter2){
		c = converter2;
	}
	if(c){
		c->DrawWave(cr, d, show_s_sec, show_t_sec, wave_y_max, select_Hz);
	}

	return FALSE;
}


static void button_test_clicked(GtkButton *button, gpointer data)
{
	if(!test_window){
		GtkWidget *vbox;

		test_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

		gtk_window_set_title(GTK_WINDOW(test_window), "sound_analyse");
		gtk_widget_set_size_request(test_window, 1200, 800);
		g_signal_connect(test_window, "destroy", G_CALLBACK(test_destroy), NULL);

		vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
		gtk_container_add(GTK_CONTAINER(test_window), vbox);

		test_drawarea = gtk_drawing_area_new();
		g_signal_connect(test_drawarea, "draw", G_CALLBACK(test_draw_callback), NULL);
		gtk_box_pack_start(GTK_BOX(vbox), test_drawarea, TRUE, TRUE, 0);

		gtk_widget_show_all(test_window);
	}
}


static void filter_destroy(GtkWidget *window, gpointer data)
{
        gtk_main_quit();
}

static void button_addstart_clicked(GtkButton *button, gpointer data)
{
	if(select_sec < 0 || select_Hz < 0) return;
	if(!converter1) return;

	GtkTextView *text = GTK_TEXT_VIEW(data);
	GtkTextBuffer *buffer = gtk_text_view_get_buffer(text);

	int ssample = select_sec * samplefq;
	std::stringstream s;
	s << "start " << ssample << "(" << select_sec << "sec) ";
	s << select_Hz << " Hz @ 1" << std::endl;

	std::string str(s.str());
	gtk_text_buffer_insert_at_cursor(buffer, str.c_str(), str.length());
}

static void button_addend_clicked(GtkButton *button, gpointer data)
{
	if(select_sec < 0) return;
	if(!converter1) return;

	GtkTextView *text = GTK_TEXT_VIEW(data);
	GtkTextBuffer *buffer = gtk_text_view_get_buffer(text);

	int ssample = select_sec * samplefq;
	std::stringstream s;
	s << "end " << ssample << "(" << select_sec << "sec) ";
	s << select_Hz << " Hz" << std::endl;

	std::string str(s.str());
	gtk_text_buffer_insert_at_cursor(buffer, str.c_str(), str.length());
}

static void button_addseed_clicked(GtkButton *button, gpointer data)
{
	if(select_sec < 0) return;
	if(!converter1) return;

	GtkTextView *text = GTK_TEXT_VIEW(data);
	GtkTextBuffer *buffer = gtk_text_view_get_buffer(text);

	int ssample = select_sec * samplefq;
	std::stringstream s;
	s << "seed " << ssample << "(" << select_sec << "sec) ";
	s << select_Hz << " Hz" << std::endl;

	std::string str(s.str());
	gtk_text_buffer_insert_at_cursor(buffer, str.c_str(), str.length());
}

static void button_filterstart_clicked(GtkButton *button, gpointer data)
{
	if(!converter2) return;
	if(!converter1) return;

	FilterSpecConverter *fconv = dynamic_cast<FilterSpecConverter *>(converter2);
	if(!fconv) return;

	GtkTextView *text = GTK_TEXT_VIEW(data);
	GtkTextBuffer *buffer = gtk_text_view_get_buffer(text);
	GtkTextIter istart, iend;

	gtk_text_buffer_get_start_iter(buffer, &istart);
	gtk_text_buffer_get_end_iter(buffer, &iend);

	gchar *strp = gtk_text_buffer_get_text(buffer, &istart, &iend, TRUE);
	std::string str(strp);
	g_free(strp);

	fconv->ApplyFilter(str);
}

static void button_invfilterstart_clicked(GtkButton *button, gpointer data)
{
	if(!converter2) return;
	if(!converter1) return;

	FilterSpecConverter *fconv = dynamic_cast<FilterSpecConverter *>(converter2);
	if(!fconv) return;

	GtkTextView *text = GTK_TEXT_VIEW(data);
	GtkTextBuffer *buffer = gtk_text_view_get_buffer(text);
	GtkTextIter istart, iend;

	gtk_text_buffer_get_start_iter(buffer, &istart);
	gtk_text_buffer_get_end_iter(buffer, &iend);

	gchar *strp = gtk_text_buffer_get_text(buffer, &istart, &iend, TRUE);
	std::string str(strp);
	g_free(strp);

	fconv->ApplyInverseFilter(str);
}

static void filter_window_create()
{
	GtkWidget *vbox, *hbox;
	GtkWidget *text, *button;

	filter_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

	gtk_window_set_title(GTK_WINDOW(filter_window), "sound_analyse");
	gtk_widget_set_size_request(filter_window, 400, 400);
	g_signal_connect(filter_window, "destroy", G_CALLBACK(filter_destroy), NULL);

	vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
	gtk_container_add(GTK_CONTAINER(filter_window), vbox);

	text = gtk_text_view_new();
	gtk_box_pack_start(GTK_BOX(vbox), text, TRUE, TRUE, 0);

	hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
	gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, TRUE, 0);

	button = gtk_button_new_with_label("add start");
	g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_addstart_clicked), text);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	button = gtk_button_new_with_label("add seed");
	g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_addseed_clicked), text);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	button = gtk_button_new_with_label("add end");
	g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_addend_clicked), text);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	button = gtk_button_new_with_label("filter start");
	g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_filterstart_clicked), text);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	button = gtk_button_new_with_label("inverse filter start");
	g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_invfilterstart_clicked), text);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	gtk_widget_show_all(filter_window);
}



static gboolean main_button_release(GtkWidget *area, GdkEvent *event, gpointer data)
{
	GdkEventButton *event_button = (GdkEventButton *)event;
	gint width = gtk_widget_get_allocated_width(area);
	gint height = gtk_widget_get_allocated_height(area);
	double time_sec = -1;
	double select_freq = -1;
	if(converter1){
		if(event_button->x >= converter1->ax_out){
			time_sec = (event_button->x - converter1->ax_out)/(width - converter1->ax_out)*show_t_sec + show_s_sec;
		}
	}
	if(test_drawarea || filter_window){
		int min_fm = 0;
		int max_fm = 0;
		if(ch1show && ch2show && converter1 && event_button->y < height/2){
			if(event_button->y < height/2 - converter1->ax_out){
				min_fm = 0;
				if(freq_min >= 0){
					while(min_fm < converter1->GetFm() && converter1->GetOmh()[min_fm] < freq_min)
						min_fm++;
				}
				max_fm = converter1->GetFm()-1;
				if(freq_max >= 0){
					while(max_fm > 0 && converter1->GetOmh()[max_fm] > freq_max)
						max_fm--;
				}
				select_freq = converter1->GetOmh()[floor((double)(height/2 - converter1->ax_out - event_button->y)/(height/2 - converter1->ax_out)*(max_fm - min_fm + 1)) + min_fm];
			}
		}
		else if(ch1show && ch2show && converter2 && event_button->y > height/2){
		       	if(event_button->y < height - converter2->ax_out){
				min_fm = 0;
				if(freq_min >= 0){
					while(min_fm < converter2->GetFm() && converter2->GetOmh()[min_fm] < freq_min)
						min_fm++;
				}
				max_fm = converter2->GetFm()-1;
				if(freq_max >= 0){
					while(max_fm > 0 && converter2->GetOmh()[max_fm] > freq_max)
						max_fm--;
				}
				select_freq = converter2->GetOmh()[floor((double)(height - converter2->ax_out - event_button->y)/(height/2 - converter2->ax_out)*(max_fm - min_fm + 1)) + min_fm];
			}
		}
		else if(ch1show && converter1 && event_button->y < height - converter1->ax_out){
			min_fm = 0;
			if(freq_min >= 0){
				while(min_fm < converter1->GetFm() && converter1->GetOmh()[min_fm] < freq_min)
					min_fm++;
			}
			max_fm = converter1->GetFm()-1;
			if(freq_max >= 0){
				while(max_fm > 0 && converter1->GetOmh()[max_fm] > freq_max)
					max_fm--;
			}
			select_freq = converter1->GetOmh()[floor((double)(height - converter1->ax_out - event_button->y)/(height - converter1->ax_out)*(max_fm - min_fm + 1)) + min_fm];
		}
		else if(ch2show && converter2 && event_button->y < height - converter2->ax_out){
			min_fm = 0;
			if(freq_min >= 0){
				while(min_fm < converter2->GetFm() && converter2->GetOmh()[min_fm] < freq_min)
					min_fm++;
			}
			max_fm = converter2->GetFm()-1;
			if(freq_max >= 0){
				while(max_fm > 0 && converter2->GetOmh()[max_fm] > freq_max)
					max_fm--;
			}
			select_freq = converter2->GetOmh()[floor((double)(height - converter2->ax_out - event_button->y)/(height - converter2->ax_out)*(max_fm - min_fm + 1)) + min_fm];
		}
	}
	select_sec = time_sec;
	select_Hz = select_freq;
	if(play){
		if(time_sec >= 0){
			player->Pause();
			player->Play(time_sec);
		}
	}
	gtk_widget_grab_focus(main_drawarea);
	return FALSE;
}

static gboolean main_key_press(GtkWidget *area, GdkEvent *event, gpointer data)
{
	GdkEventKey *event_key = (GdkEventKey *)event;
	double max_sec = (double)maxsamples / samplefq;
	if(event_key->type == GDK_KEY_PRESS){
		bool shift = event_key->state & GDK_SHIFT_MASK;
		bool ctl = event_key->state & GDK_CONTROL_MASK;
		switch(event_key->keyval){
			case GDK_KEY_Left:
				if(shift && ctl)
					select_sec -= 1.0;
				else if(ctl)
					select_sec -= 0.1;
				else if(shift)
					select_sec -= 0.001;
				else
					select_sec -= 1.0/samplefq;
				break;
			case GDK_KEY_Right:
				if(shift && ctl)
					select_sec += 1.0;
				else if(ctl)
					select_sec += 0.1;
				else if(shift)
					select_sec += 0.001;
				else
					select_sec += 1.0/samplefq;
				break;
			case GDK_KEY_Up:
				if(converter1){
					int s_fm = 0;
					while(s_fm++ < converter1->GetFm() && converter1->GetOmh()[s_fm] < select_Hz);
					s_fm++;
					if(s_fm >= converter1->GetFm()) s_fm = converter1->GetFm() - 1;
					select_Hz = converter1->GetOmh()[s_fm];
				}
				break;
			case GDK_KEY_Down:
				if(converter1){
					int s_fm = 0;
					while(s_fm++ < converter1->GetFm() && converter1->GetOmh()[s_fm] < select_Hz);
					s_fm--;
					if(s_fm < 0) s_fm = 0;
					select_Hz = converter1->GetOmh()[s_fm];
				}
				break;
			case GDK_KEY_Home:
				select_sec = 0;
				break;
			case GDK_KEY_End:
				select_sec = max_sec;
				break;
		}
	}
	if(select_sec < 0) select_sec = 0;
	if(select_sec > max_sec) select_sec = max_sec;
	if(select_sec < show_s_sec || select_sec > show_s_sec + show_t_sec){
		show_s_sec = converter1->ScrollToTime(select_sec);
	}
	gtk_widget_queue_draw(main_drawarea);
	return FALSE;
}

void RunGUI(int maxsamples, int samplefq, bool isfilter = false)
{
	GtkWidget *window, *vbox, *hbox;
	GtkWidget *hscale, *button;
	GtkWidget *tbutton, *label, *entry;
	GtkAdjustment *adj1;
	GtkListStore *list_store;
	GtkCellRenderer *column;
	GtkWidget *combo;

        window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

        gtk_window_set_title(GTK_WINDOW(window), "sound_analyse");
        gtk_widget_set_size_request(window, 1200, 800);
        g_signal_connect(G_OBJECT(window), "destroy", G_CALLBACK(destroy), NULL);
        g_signal_connect(G_OBJECT(window), "delete-event", G_CALLBACK(delete_event), NULL);

	vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
	gtk_container_add(GTK_CONTAINER(window), vbox);

        main_drawarea = gtk_drawing_area_new();
        g_signal_connect(main_drawarea, "draw", G_CALLBACK(draw_callback), NULL);
	gtk_widget_add_events(main_drawarea, GDK_BUTTON_RELEASE_MASK | GDK_BUTTON_PRESS_MASK);
	g_signal_connect(main_drawarea, "button-release-event", G_CALLBACK(main_button_release), NULL);
	gtk_widget_add_events(main_drawarea, GDK_KEY_PRESS_MASK);
	gtk_widget_set_can_focus(main_drawarea, TRUE);
	gtk_widget_grab_focus(main_drawarea);
	g_signal_connect(main_drawarea, "key-press-event", G_CALLBACK(main_key_press), NULL);
	gtk_box_pack_start(GTK_BOX(vbox), main_drawarea, TRUE, TRUE, 0);

	adj1 = gtk_adjustment_new(0, 0, (double)maxsamples/samplefq, 1, 5, 2);
	hscale = gtk_scale_new(GTK_ORIENTATION_HORIZONTAL, GTK_ADJUSTMENT(adj1));
	gtk_scale_set_digits(GTK_SCALE(hscale), 1);
	gtk_scale_set_value_pos(GTK_SCALE(hscale), GTK_POS_TOP);
	gtk_scale_set_draw_value(GTK_SCALE(hscale), TRUE);
        g_signal_connect(G_OBJECT(adj1), "value-changed", G_CALLBACK(value_changed_callback), NULL);
	gtk_box_pack_start(GTK_BOX(vbox), hscale, FALSE, TRUE, 0);

	hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

	gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, TRUE, 0);
	button = gtk_button_new_with_label("play");
        g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_play_clicked), NULL);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);
	button = gtk_button_new_with_label("stop");
        g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_stop_clicked), NULL);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	label = gtk_label_new("Display Control");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

	tbutton = gtk_toggle_button_new_with_label("Ch1 Display");
	gtk_toggle_button_set_mode(GTK_TOGGLE_BUTTON(tbutton), TRUE);
	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(tbutton), TRUE);
	g_signal_connect(tbutton, "toggled", G_CALLBACK(toggle_button_toggled), &ch1show);
	gtk_box_pack_start(GTK_BOX(hbox), tbutton, FALSE, FALSE, 0);
	if(converter2){
		tbutton = gtk_toggle_button_new_with_label("Ch2 Display");
		gtk_toggle_button_set_mode(GTK_TOGGLE_BUTTON(tbutton), TRUE);
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(tbutton), TRUE);
		g_signal_connect(tbutton, "toggled", G_CALLBACK(toggle_button_toggled), &ch2show);
		gtk_box_pack_start(GTK_BOX(hbox), tbutton, FALSE, FALSE, 0);
	}
	else{
		ch2show = false;
	}

	label = gtk_label_new("x range(sec)");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

	entry = gtk_entry_new();
	gtk_entry_set_text(GTK_ENTRY(entry), "5.0");
	g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &show_t_sec);
	gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);

	label = gtk_label_new("pitch A440(Hz)");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

	entry = gtk_entry_new();
	gtk_entry_set_text(GTK_ENTRY(entry), "440.0");
	g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &A440);
	gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);

	list_store = gtk_list_store_new(1, G_TYPE_STRING);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "None", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "Ces_Dur(♭7)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "Ges_Dur(♭6)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "Des_Dur(♭5)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "As_Dur(♭4)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "Es_Dur(♭3)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "B_Dur(♭2)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "F_Dur(♭1)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "C_Dur(0)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "G_Dur(♯1)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "D_Dur(♯2)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "A_Dur(♯3)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "E_Dur(♯4)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "H_Dur(♯5)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "Fis_Dur(♯6)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "Cis_Dur(♯7)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "as_Moll(♭7)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "es_Moll(♭6)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "b_Moll(♭5)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "f_Moll(♭4)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "c_Moll(♭3)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "g_Moll(♭2)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "d_Moll(♭1)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "a_Moll(0)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "e_Moll(♯1)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "h_Moll(♯2)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "fis_Moll(♯3)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "cis_Moll(♯4)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "gis_Moll(♯5)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "dis_Moll(♯6)", -1);
	gtk_list_store_insert_with_values(list_store, NULL, -1, 0, "ais_Moll(♯7)", -1);

	combo = gtk_combo_box_new_with_model(GTK_TREE_MODEL(list_store));
	g_object_unref(list_store);

	column = gtk_cell_renderer_text_new();
	gtk_cell_layout_pack_start(GTK_CELL_LAYOUT(combo), column, TRUE);
	gtk_cell_layout_set_attributes(GTK_CELL_LAYOUT(combo), column, "text", 0, NULL);

	gtk_combo_box_set_active(GTK_COMBO_BOX(combo), 0);
	g_signal_connect(combo, "changed", G_CALLBACK(combo_changed), NULL);

	label = gtk_label_new("key note");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);
	
	gtk_box_pack_start(GTK_BOX(hbox), combo, FALSE, FALSE, 0);

	button = gtk_button_new_with_label("spectrogram");
        g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_spectro_clicked), NULL);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	tbutton = gtk_toggle_button_new_with_label("Filterd Disp");
	gtk_toggle_button_set_mode(GTK_TOGGLE_BUTTON(tbutton), TRUE);
	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(tbutton), FALSE);
	g_signal_connect(tbutton, "toggled", G_CALLBACK(toggle_button_toggled), &filtshow);
	gtk_box_pack_start(GTK_BOX(hbox), tbutton, FALSE, FALSE, 0);

	button = gtk_button_new_with_label("wave");
        g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_wave_clicked), NULL);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	label = gtk_label_new("fq min(Hz)");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

	entry = gtk_entry_new();
	gtk_entry_set_text(GTK_ENTRY(entry), "-1");
	g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &freq_min);
	gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);
	
	label = gtk_label_new("fq max(Hz)");
	gtk_box_pack_start(GTK_BOX(hbox), label, FALSE, FALSE, 0);

	entry = gtk_entry_new();
	gtk_entry_set_text(GTK_ENTRY(entry), "-1");
	g_signal_connect(entry, "activate", G_CALLBACK(entry_activate), &freq_max);
	gtk_box_pack_start(GTK_BOX(hbox), entry, FALSE, FALSE, 0);
	
	button = gtk_button_new_with_label("unit detail");
        g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(button_test_clicked), NULL);
	gtk_box_pack_start(GTK_BOX(hbox), button, FALSE, FALSE, 0);

	gtk_widget_show_all(window);

	//////////////////////////////////////////////////////////////

	timer = g_timeout_add(100, GSourceFunc(timer_callback), adj1);

	if(isfilter)
		filter_window_create();

        gtk_main();
}

void PrintUsage(char *my)
{
	printf("usage: %s (options) input.wav\n", my);
	printf("options\n");
	printf("  --mono: force mono\n");
	printf("  --start [time]: skip first wav samples\n");
	printf("  --length [time]: analyse only specify length\n");
	printf("  --filter: filter mode\n");
	printf("time option\n");
	printf("  10.05   means 10.05sec\n");
	printf("  1min5sec or 1h10m5.0sec can be acceptable\n");
}

double GetTimeString(char *input_str)
{
	std::string str(input_str);
	std::smatch match;
	std::regex re(R"(((\d+)h(our)?)?((\d+)m(in)?)?((\d+\.?\d*)(s(ec)?)?)?)");

	if(regex_match(str, match, re)){
		double hour, min, sec;
		std::stringstream(match[2].str()) >> hour;
		std::stringstream(match[5].str()) >> min;
		std::stringstream(match[8].str()) >> sec;
		return hour*3600+min*60+sec;
	}
	else
		return -1;
}

int main(int argc, char *argv[])
{
	if(argc < 2){
		PrintUsage(argv[0]);
		return 0;
	}

        gtk_init(&argc, &argv);

	struct option longopts[] = {
		{ "mono", 	no_argument,       	NULL,	'm' },
		{ "filter", 	no_argument,       	NULL,	'f' },
		{ "length",	required_argument, 	NULL, 	'l' },
		{ "start", 	required_argument, 	NULL, 	's' },
		{ 0,       	0,                	0,	0   },
	};

	int opt;
	int longindex;

	bool ismono = false;
	bool isfilter = false;
	double wavlength = -1;
	double wavstart = -1;

	while((opt = getopt_long_only(argc, argv, "", longopts, &longindex)) != -1){
		switch(opt){
			case 'm':
				ismono = true;
				break;
			case 'f':
				isfilter = true;
				break;
			case 'l':
				wavlength = GetTimeString(optarg);
				break;
			case 's':
				wavstart = GetTimeString(optarg);
				break;
			default:
				PrintUsage(argv[0]);
				return 1;
		}
	}
	if(wavstart > 0){
		printf("analysis start on %f sec of wav file\n", wavstart);

	}
	if(wavlength > 0){
		printf("analysis limits %f sec length\n", wavlength);
	}
	
	if(wavstart < 0) wavstart = 0;


	if(optind == argc){
		PrintUsage(argv[0]);
		return 1;
	}

	if(isfilter){
		auto wav1 = WaveData(argv[optind], wavstart, wavlength);
		if(wav1.loaded_samples <= 0){
			printf("wav file open error\n");
			return 1;
		}
		SpecConverter converter_1(wav1, WaveChannel::Mono); 
		FilterSpecConverter converter_2(wav1, WaveChannel::Mono); 
		Player player_1(converter_2);

		converter1 = &converter_1;
		converter2 = &converter_2;
		player = &player_1;

		maxsamples = wav1.loaded_samples;
		samplefq = wav1.samplerate;


		RunGUI(maxsamples, samplefq, true);
	}
	else if(ismono){
		auto wav1 = WaveData(argv[optind], wavstart, wavlength);
		if(wav1.loaded_samples <= 0){
			printf("wav file open error\n");
			return 1;
		}
		SpecConverter converter_1(wav1, WaveChannel::Mono); 
		Player player_1(converter_1);

		converter1 = &converter_1;
		player = &player_1;

		maxsamples = wav1.loaded_samples;
		samplefq = wav1.samplerate;


		RunGUI(maxsamples, samplefq);
	}
	else{
		auto wav1 = WaveData(argv[optind], wavstart, wavlength);
		if(wav1.loaded_samples <= 0){
			printf("wav file open error\n");
			return 1;
		}
		if(wav1.right){
			SpecConverter converter_1(wav1, WaveChannel::Left); 
			SpecConverter converter_2(wav1, WaveChannel::Right); 
			Player player_1(converter_1, converter_2);

			converter1 = &converter_1;
			converter2 = &converter_2;
			player = &player_1;

			maxsamples = wav1.loaded_samples;
			samplefq = wav1.samplerate;


			RunGUI(maxsamples, samplefq);
		}
		else {
			SpecConverter converter_1(wav1, WaveChannel::Left); 
			Player player_1(converter_1);

			converter1 = &converter_1;
			player = &player_1;

			maxsamples = wav1.loaded_samples;
			samplefq = wav1.samplerate;


			RunGUI(maxsamples, samplefq);
		}
	}

	return 0;
}
