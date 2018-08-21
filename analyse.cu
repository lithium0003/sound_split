#include "analyse.hpp"
#include <helper_cuda.h>

extern bool done;

//コンストラクタ 初期設定
DetectionUnits::DetectionUnits(int SampleFq, float base_fq, float max_fq)
	:nz(SampleFq), omh_base(base_fq), omh_max(max_fq),
	ih(NULL), clen(NULL), sdds(NULL), sddc(NULL), spec(NULL), 
	aasc(NULL),
	spec2(NULL), fspec(NULL), mask(NULL), comh(NULL), tidx(NULL),
	ws(NULL), wc(NULL), fm(0), omh(0)
{
	printf("SampleFreq=%d\n", SampleFq);
	printf("minFreq=%f\n", base_fq);
	printf("maxFreq=%f\n", max_fq);

	nnz1=t*nz;
	h=1.0/nz;
	nnz2=interp*nnz1;
	dt1=1.0/nz;
	dt2=1.0/(nz*interp);
	gm=-1;

	cutlen.push_back(nnz1/omh_base);
	omh.push_back((float)nnz1/cutlen[0]);
	om.push_back(omh[0]*2*M_PI);

	init_detection_unit();
	init_cutwaves();

	checkCudaErrors(cudaStreamCreate(&stream[0]));
	checkCudaErrors(cudaStreamCreate(&stream[1]));

	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 8*1024));
}

// 信号検出ユニットの設定周波数計算
void DetectionUnits::init_detection_unit()
{
	int i=0;
	printf("min on unit %d=%fHz\n", i, omh[i]);
	while(omh[i++] < omh_max)
	{
		if(gm < 0){
			cutlen.push_back(cutlen[i-1]/f);
			omh.push_back((float)nnz1/cutlen[i]);
			om.push_back(omh[i]*2*M_PI);
			if(cutlen[i] < start_interp){
				gm = i+1;
				printf("switch to spline on unit %d=%fHz\n", i, omh[i]);
			}
		}
		else{
			if(gm == i){
				cutlen.push_back(cutlen[i-1]*interp/f);
			}
			else{
				cutlen.push_back(cutlen[i-1]/f);
			}
			omh.push_back((float)nnz2/cutlen[i]);
			om.push_back(omh[i]*2*M_PI);
		}
	}
	fm = i--;
	printf("max on unit %d=%fHz\n", i, omh[i]);

	specbuflen = 512 * 1024 * 1024 / (sizeof(float) * fm * interp) * interp;
}

//切り出し波の設定
void DetectionUnits::init_cutwaves()
{
	checkCudaErrors(cudaMallocManaged(&ws, fm*sizeof(float*), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMallocManaged(&wc, fm*sizeof(float*), CU_MEM_ATTACH_GLOBAL));

	int j=0;
	for(auto cl: cutlen){
		checkCudaErrors(cudaMallocManaged(&ws[j], (cl+1)*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMallocManaged(&wc[j], (cl+1)*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		j++;
	}

	j=0;
	for(auto cl: cutlen){
		if(j < gm){
			for(int i=1; i<=cl; i++){
				ws[j][i] = sin(om[j] * dt1 * i);
				wc[j][i] = cos(om[j] * dt1 * i);
			}
		}
		else {
			for(int i=1; i<=cl; i++){
				ws[j][i] = sin(om[j] * dt2 * i);
				wc[j][i] = cos(om[j] * dt2 * i);
			}
		}
		j++;
	}
}


DetectionUnits::~DetectionUnits()
{
	while(callcount > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));

	for(int i = 0; i < fm; i++){
		cudaFree(ws[i]);
		cudaFree(wc[i]);
	}
	cudaFree(ws);
	ws = NULL;
	cudaFree(wc);
	wc = NULL;

	cudaFree(aasc);
	aasc = NULL;
	cudaFree(spec);
	spec = NULL;
	cudaFree(fspec);
	fspec = NULL;
	cudaFree(spec2);
	spec2 = NULL;
	cudaFree(mask);
	mask = NULL;
	cudaFree(comh);
	comh = NULL;
	cudaFree(tidx);
	tidx = NULL;
	cudaFree(sddc);
	sddc = NULL;
	cudaFree(sdds);
	sdds = NULL;
	cudaFree(clen);
	clen = NULL;
	cudaFree(ih);
	ih = NULL;
	
	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
}


struct params
{
	const float *z;
	const long zlen;
	const float *s;
	const int sstart;
	const int slope;
	const int *cutlen;
	const int gm;
	const int fm;
	const float **ws;
	const float **wc;
	int *ih;
	float *sdds;
	float *sddc;
	float *spec;
	float *aasc;
	int *op;
	int *ip;
};

#define thread_N 32
#define thread_N2 1024

__device__ void process_wave(float *sdds, float *sddc, float *spec, float *aasc, const float **ws, const float **wc, int idx, int ih, int c, int fm, int outp, float s_ii, float s_ik)
{
	float sds = sdds[idx] + ws[idx][ih]*s_ii - ws[idx][ih]*s_ik;
	float sdc = sddc[idx] + wc[idx][ih]*s_ii - wc[idx][ih]*s_ik;
	sdds[idx] = sds;
	sddc[idx] = sdc;
	float sads = 2.0 * sds / c;
	float sadc = 2.0 * sdc / c;
	float adss = sads * ws[idx][ih];
	float adsc = sads * wc[idx][ih];
	float adcc = sadc * wc[idx][ih];
	float adcs = sadc * ws[idx][ih];
	float aadc = adss + adcc;
	float aads = adsc - adcs;
	spec[outp*fm+idx] = sqrtf(aads * aads + aadc * aadc);
	aasc[outp*fm+idx] = atan2f(aads, aadc);
}

__device__ void process_wave2(float *sdds, float *sddc, const float **ws, const float **wc, int idx, int ih, float s_ii, float s_ik)
{
	float sds = sdds[idx] + ws[idx][ih]*s_ii - ws[idx][ih]*s_ik;
	float sdc = sddc[idx] + wc[idx][ih]*s_ii - wc[idx][ih]*s_ik;
	sdds[idx] = sds;
	sddc[idx] = sdc;
}

__device__ float slope_data(const float *data, int i, int len, int slope)
{
	if(i < 0 || i >= len) return 0;
	if(slope != 0 && i < slope) return data[i] * (float)i/slope;
	return data[i];
}

__device__ float limit_data(const float *data, int i, int len)
{
	if(i < 0 || i >= len) return 0;
	return data[i];
}

__global__ void kernel3(int plen, struct params p)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= p.fm){
	       	return;
	}

	int zlen = p.zlen;
	int slen = zlen / interp;
	int sstart = p.sstart;
	int zstart = sstart * interp;
	int slope = p.slope;
	int zslope = p.slope * interp;

	bool zflag = (idx < p.gm);
	
	int c = p.cutlen[idx];
	int ih = p.ih[idx];
	int outp = 0;
	int ic = 1;
	for(; ic < zlen && outp < plen/interp; ic++){
		int ii = (zflag)? ic/interp: ic;
		int ik = ii - c;
		if(ic % interp == 1){
			float s_ii = slope_data((zflag)? p.s: p.z, ii, (zflag)? slen: zlen, slope*((zflag)?1:interp));
			float s_ik = slope_data((zflag)? p.s: p.z, ik, (zflag)? slen: zlen, slope*((zflag)?1:interp));
			if(ih <= 0)
				ih = c;
			if(ic < zstart){
				process_wave2(p.sdds, p.sddc, p.ws, p.wc, idx, ih, s_ii, s_ik);
			}
			else{
				process_wave(p.sdds, p.sddc, p.spec, p.aasc, p.ws, p.wc, idx, ih, c, p.fm, outp++, s_ii, s_ik);
			}
			ih--;
		}
		else if(!zflag){
			float z_ii = slope_data(p.z, ii, zlen, zslope);
			float z_ik = slope_data(p.z, ik, zlen, zslope);
			if(ih <= 0)
				ih = c;
			process_wave2(p.sdds, p.sddc, p.ws, p.wc, idx, ih, z_ii, z_ik);
			ih--;
		}
	}
	p.ih[idx] = ih;

	if(idx != 0) return;
	*(p.ip) = ic;
	*(p.op) = outp;
}

__global__ void kernel4(int sindex, int plen, struct params p)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= p.fm) return;

	int zlen = p.zlen;
	int slen = zlen / interp;

	bool zflag = (idx < p.gm);
	int c = p.cutlen[idx];
	int ih = p.ih[idx];

	int outp = 0;
	int ic;
	for(ic = sindex; ic < zlen && ic < sindex + plen; ic++){
		int ii = (zflag)? ic/interp: ic;
		int ik = ii - c;
		if(ic % interp == 1){
			float s_ii = limit_data((zflag)? p.s: p.z, ii, (zflag)? slen: zlen);
			float s_ik = limit_data((zflag)? p.s: p.z, ik, (zflag)? slen: zlen);
			if(ih <= 0)
				ih = c;
			process_wave(p.sdds, p.sddc, p.spec, p.aasc, p.ws, p.wc, idx, ih, c, p.fm, outp++, s_ii, s_ik);
			ih--;
		}
		else if(!zflag){
			float z_ii = limit_data(p.z, ii, zlen);
			float z_ik = limit_data(p.z, ik, zlen);
			if(ih <= 0)
				ih = c;
			process_wave2(p.sdds, p.sddc, p.ws, p.wc, idx, ih, z_ii, z_ik);
			ih--;
		}
	}
	p.ih[idx] = ih;

	if(idx != 0) return;
	*(p.ip) = ic;
	*(p.op) = outp;

}

__device__ float mainlobe(float Hz, float peakHz, float peakValue)
{
	float f = log10f(Hz) - log10f(peakHz);
	float a = -1e4;
	return a*f*f + peakValue+0.1;
}

__device__ float lowerslope(float Hz, float peakHz, float b)
{
	float f1 = log10f(fabsf(Hz-peakHz)/peakHz);
	float f2 = log10f(Hz);
	return -17*f1 + 14*f2 + b;
}

__device__ float higherslope(float Hz, float peakHz, float b)
{
	float f1 = log10f(fabsf(Hz-peakHz)/peakHz);
	float f2 = log10f(Hz);
	return -19*f1 + 22*f2 + b;
}

#define noiseLevel -70

__device__ void findmask(float *mask, float *result, float *src, int *idx, int fcont, float *omh, int fm)
{
	int maxidx = 0;
	int count = 0;
	for(int i = (0 < idx[fcont]-3)? idx[fcont]-3 : 0; i < ((idx[fcont]+4 < fm)? idx[fcont]+4: fm); i++){
		if(src[i] == src[idx[fcont]]){
			maxidx = i;
			count++;
		}
	}
	if(count > 0)
		maxidx /= count;
	else
		maxidx = idx[fcont];

	float Hz = omh[maxidx];
	float value = src[maxidx];
	float valuedB = 20*log10f(value);
	float lowerPoint = Hz - Hz*0.06;
	float higherPoint = Hz + Hz*0.06;
	float b1 = mainlobe(lowerPoint, Hz, valuedB) - lowerslope(lowerPoint, Hz, 0);
	float b2 = mainlobe(higherPoint, Hz, valuedB) - higherslope(higherPoint, Hz, 0);

	for(int j = fcont; j >= 0; j--){
		int i = idx[j];
		float fq = omh[i];
		float vmask = value;
		float vmaskdB = valuedB;
	
		if(i == maxidx){
			result[i] = src[i];
		}
		if(i == maxidx){
			vmask = value;
		}
		else if (maxidx - 5 < i && i < maxidx + 5){
			vmask = value;
		}
		else if(fq < lowerPoint){
			vmaskdB = lowerslope(fq, Hz, b1);
			vmask = powf(10, vmaskdB/20);
		}
		else if(fq > higherPoint){
			vmaskdB = higherslope(fq, Hz, b2);
			vmask = powf(10, vmaskdB/20);
		}
		else{
			vmaskdB = mainlobe(fq, Hz, valuedB);
			vmask = powf(10, vmaskdB/20);
		}

		//if(mask[i] < vmask)
		//	mask[i] = vmask;
		mask[i] += vmask;
	}

}

__device__ void selection_sort(float *data, int *idx, int left, int right)
{
	for(int i = left; i <= right; i++){
		int minidx = i;
		float minvalue = data[idx[minidx]];
		
		for(int j = i+1; j <= right; j++){
			if(data[idx[j]] < minvalue){
				minidx = j;
				minvalue = data[idx[j]];
			}
		}
		if(minidx != i){
			int t = idx[minidx];
			idx[minidx] = idx[i];
			idx[i] = t;
		}
	}
}

__device__ void quick_sort(float *data, int *idx, int left, int right, int depth)
{
	if(left >= right) return;
	if(right - left < 4 || depth > 32){
		selection_sort(data, idx, left, right);
		return;
	}

	int leftp = left;
	int rightp = right;
	int pivotp = (right + left)/2;
	if(idx[pivotp] <0)
		printf("error\n");
	float pivot = data[idx[pivotp]];

	while(leftp <= rightp){
		if(idx[leftp] < 0)
			printf("error\n");
		if(idx[rightp] < 0)
			printf("error\n");
		while(data[idx[leftp]] < pivot && leftp < right){
			leftp++;
			if(idx[leftp] < 0)
				printf("error\n");
		}
		while(data[idx[rightp]] > pivot && left < rightp){
			rightp--;
			if(idx[rightp] < 0)
				printf("error\n");
		}
		if(leftp <= rightp){
			int t = idx[leftp];
			idx[leftp] = idx[rightp];
			idx[rightp] = t;
			leftp++;
			rightp--;
		}
	}

	if(idx[leftp] < 0)
		printf("error\n");
	if(idx[rightp] < 0)
		printf("error\n");
	if(rightp > left){
		quick_sort(data, idx, left, rightp, depth+1);
	}
	if(leftp < right){
		quick_sort(data, idx, leftp, right, depth+1);
	}
}


__global__ void kernel5(float *spec, int *tidx, float *fspec, float *mask, float *comh, int len, int fm)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= len) return;

	spec = &spec[idx*fm];
	tidx = &tidx[idx*fm];
	fspec = &fspec[idx*fm];
	mask = &mask[idx*fm];

	const float cutoff = 10000;
	const float noise = powf(10, noiseLevel/20);

	for(int i = 0; i < fm; i++){
		mask[i] = 0;
		fspec[i] = 0;
		tidx[i] = -1;
	}
	int c = 0;
	for(int i = 0; i < fm; i++){
		if(comh[i] > cutoff) break;
		float value = spec[i];
		if(value <= noise) continue;
		tidx[c++] = i;
	}
	if(c == 0) return;
	if(c > 0)
		quick_sort(spec, tidx, 0, c-1, 0);

	for(int i = c-1; i >= 0; i--){
		int max_idx = tidx[i];
		float max_value = spec[max_idx];
		if(max_value <= mask[max_idx]) continue;
		
		findmask(mask, fspec, spec, tidx, i, comh, fm);
	}
}
	
	
//解析を実施する
int DetectionUnits::AnalyzeData(
		std::vector<float> &ret_spec, 
		std::vector<float> &ret_aasc, 
		int offset, 
		int len, 
		const float *s, 
		int slen, 
		const float *z)
{
	ret_spec.resize(len*fm);
	ret_aasc.resize(len*fm);
	memset(ret_spec.data(), 0, sizeof(float)*len*fm);
	memset(ret_aasc.data(), 0, sizeof(float)*len*fm);

	if(offset > slen) return 0;
	if(offset + len > slen) len = slen - offset;
	callcount++;

	int c_buf = cutlen[0];

	int prepad = c_buf*3;
	int slope = c_buf*2;

	if(!ih)
		checkCudaErrors(cudaMallocManaged(&ih, fm*sizeof(int), CU_MEM_ATTACH_GLOBAL));
	if(!clen){
		checkCudaErrors(cudaMallocManaged(&clen, fm*sizeof(int), CU_MEM_ATTACH_GLOBAL));
		memcpy(clen, cutlen.data(), fm*sizeof(int));
	}
	if(!comh){
		checkCudaErrors(cudaMallocManaged(&comh, fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		memcpy(comh, omh.data(), fm*sizeof(float));
	}
	memcpy(ih, cutlen.data(), fm*sizeof(int));

	if(!sdds)
		checkCudaErrors(cudaMallocManaged(&sdds, fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	if(!sddc)
		checkCudaErrors(cudaMallocManaged(&sddc, fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	memset(sdds, 0, fm*sizeof(float));
	memset(sddc, 0, fm*sizeof(float));

	if(!spec)
		checkCudaErrors(cudaMallocManaged(&spec, specbuflen*fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	if(!aasc)
		checkCudaErrors(cudaMallocManaged(&aasc, specbuflen*fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));

	int *op, *ip;
	checkCudaErrors(cudaMallocManaged(&op, sizeof(int), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMallocManaged(&ip, sizeof(int), CU_MEM_ATTACH_GLOBAL));
	*op = 0;
	*ip = 0;

	if(prepad > offset)
		prepad = offset;
	if(prepad < c_buf)
		slope = 0;
	else
		slope = prepad - cutlen[0];
	int fixoffset = offset - prepad;
	int fixlen = prepad + len;

	printf("AnalyzeData %d\n", fixlen*interp);
	struct params p = {&z[fixoffset*interp], fixlen*interp, &s[fixoffset], prepad, slope, clen, gm, fm, (const float **)ws, (const float **)wc, ih, sdds, sddc, spec, aasc, op, ip};
	int idx = 0;
	while(*ip < fixlen*interp){
		int plen = (*ip + specbuflen < fixlen*interp)? specbuflen: fixlen*interp - *ip;
		if(*ip > 0){
			kernel4<<<ceil((float)fm/thread_N), thread_N, 0, stream[0]>>>(*ip, plen, p);
		}
		else{
			kernel3<<<ceil((float)fm/thread_N), thread_N, 0, stream[0]>>>(plen, p);
		}
		cudaStreamSynchronize(stream[0]);

		int lenp = *op;
		if(lenp <= 0){
			printf("return length is 0\n");
			break;
		}
		memcpy(&ret_spec[idx*fm], spec, lenp*fm*sizeof(float));
		memcpy(&ret_aasc[idx*fm], aasc, lenp*fm*sizeof(float));
		idx += lenp;
	}
	printf("AnalyzeData %d end\n", fixlen*interp);

	cudaFree(op);
	cudaFree(ip);

	callcount--;
	return idx;
}


int DetectionUnits::FilterData(
		const std::vector<float> &in_spec, 
		std::vector<float> &ret_fspec)
{
	int len = in_spec.size()/fm;
	ret_fspec.resize(len*fm);
	memset(ret_fspec.data(), 0, sizeof(float)*len*fm);

	callcount++;

	const int buflen = 64 * 1024;

	if(!spec2)
		checkCudaErrors(cudaMallocManaged(&spec2, buflen*fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	if(!fspec)
		checkCudaErrors(cudaMallocManaged(&fspec, buflen*fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	if(!mask)
		checkCudaErrors(cudaMallocManaged(&mask, buflen*fm*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	if(!tidx)
		checkCudaErrors(cudaMallocManaged(&tidx, buflen*fm*sizeof(int), CU_MEM_ATTACH_GLOBAL));

	printf("FilterData %d\n", len);
	int idx = 0;
	while(idx < len){
		int plen = (idx + buflen < len)? buflen: len - idx;
		printf("plen2 %d\n", plen);
		memcpy(spec2, &in_spec[idx*fm], plen*fm*sizeof(float));
		kernel5<<<ceil((float)plen/thread_N2), thread_N2, 0, stream[0]>>>(spec2, tidx, fspec, mask, comh, plen, fm);
		cudaStreamSynchronize(stream[0]);

		memcpy(&ret_fspec[idx*fm], fspec, plen*fm*sizeof(float));
		idx += plen;
	}
	printf("FilterData %d end\n", len);

	callcount--;
	return idx;
}
