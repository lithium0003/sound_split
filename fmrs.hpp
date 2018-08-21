#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <string>
#include <limits>
#include <cmath>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <map>
#include <list>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

#define interp 10     //スプライン補間倍数
#define spec_call_sec 5.0

#include "analyse.hpp"
#include "wave.hpp"


enum class Tonality {
	None = 0,
	Ces_Dur,
	Ges_Dur,
	Des_Dur,
	As_Dur,
	Es_Dur,
	B_Dur,
	F_Dur,
	C_Dur,
	G_Dur,
	D_Dur,
	A_Dur,
	E_Dur,
	H_Dur,
	Fis_Dur,
	Cis_Dur,
	as_Moll,
	es_Moll,
	b_Moll,
	f_Moll,
	c_Moll,
	g_Moll,
	d_Moll,
	a_Moll,
	e_Moll,
	h_Moll,
	fis_Moll,
	cis_Moll,
	gis_Moll,
	dis_Moll,
	ais_Moll,
};


