#pragma once

#include "fmrs.hpp"

class DetectionUnits
{
	private:
		int callcount = 0;
		int nz;  // サンプリング周波数Fs

		//データスプライン補間開始閾値
		//切り出しのサンプル数がこの長さ以下になれば
		//スプライン補間を開始する
		const int start_interp=10000;

		float omh_base;    //最低設定周波数
		float omh_max;     //最高周波数の目安

		int nnz1;          //t周期数のデータ数
		int nnz2;          //補間後のt周期数のデータ数
		float h;           //サンプリング間隔Ts
		float dt1;         //低設定周波数の切り出し波のサンプリング間隔
		float dt2;         //高設定周波数の切り出し波のサンプリング間隔
		int gm;            //データのスプライン補間開始する信号検出ユニット番号
		std::vector<float> om;
		std::vector<int> cutlen;  //設定周波数の切り出し波の長さ(データ数)

		float **ws;        //切り出し波ws
		float **wc;        //切り出し波wc

		long specbuflen;
		int *ih;
		int *clen;
	        float *sdds;
	       	float *sddc;
        	float *spec;
        	float *fspec;
        	float *spec2;
        	float *mask;
		float *comh;
		int *tidx;
	        float *aasc;

		cudaStream_t stream[2];

		// 信号検出ユニットの設定周波数計算
		void init_detection_unit();
		//切り出し波の設定
		void init_cutwaves();

	public:
		const int t=16;        // 周期数T
		const float f = 1.005;  //設定周波数増加率σ
		std::vector<float> omh;  //設定周波数
		int fm;             //信号検出ユニットの数

		DetectionUnits(int SampleFq, float base_fq, float max_fq);
		~DetectionUnits();
		int AnalyzeData(
				std::vector<float> &ret_spec, 
				std::vector<float> &ret_aasc, 
				int offset, 
				int len, 
				const float *s, 
				int slen, 
				const float *z);
		int FilterData(
				const std::vector<float> &in_spec, 
				std::vector<float> &ret_fspec);

};
