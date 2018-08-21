#include "wave.hpp"
#include <helper_cuda.h>

#define thread_N 1024

__global__ void kernel1(float *d, const float *src, int samplerate, long srclen)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	d[idx] = 0;

	if(idx < 1) return;
	if(idx > srclen-2) return;

	float sig0 = (src[idx] - src[idx-1])*samplerate;
	float sig1 = (src[idx+1] - src[idx])*samplerate;

	d[idx] = 3*(sig1 - sig0)*samplerate;
}

__global__ void kernel2(float *dst, const float *src, const float *mm, int samplerate, long srclen)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < 1){
		dst[0] = 0;
	}
	else if(idx < srclen){
		int xi = interp * (idx-1);

		float m0 = mm[idx-1] / 6 * samplerate;
		float m1 = mm[idx] / 6 * samplerate;
		float f0 = src[idx-1] * samplerate - mm[idx-1] / samplerate / 6;
		float f1 = src[idx] * samplerate - mm[idx] / samplerate / 6;
		for(int i = 1; i <= interp; i++){
			float fi = (float)(idx-1) / samplerate;
			float ff = (float)idx / samplerate;
			float xii = (float)(xi + i) / interp / samplerate;
			dst[xi+i] = m0 * powf((ff - xii), 3)
					+ m1 * powf((xii - fi), 3)
					+ f0 * (ff - xii)
					+ f1 * (xii - fi);
		}
	}
}

void WaveData::InterpolateSpline(float **dst, const float *src, cudaStream_t stream)
{
	if(!*dst){
		interp_samples = loaded_samples * interp + 1;
		checkCudaErrors(cudaMallocManaged(dst, interp_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	}
	else{
		assert(interp_samples == loaded_samples * interp + 1);
	}

	const float lambda = 0.5;
	const float mu = 0.5;

	float a =2.0;
	float b = lambda;
	float c = mu;
	float *d;
	checkCudaErrors(cudaMallocManaged(&d, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));

	kernel1<<<ceil((float)loaded_samples/thread_N), thread_N, 0, stream>>>(d, src, samplerate, loaded_samples);
	cudaStreamSynchronize(stream);

	std::vector<float> hh;
	std::vector<float> kk;
	hh.push_back(d[0]/a);
	hh.push_back(d[1]/a);
	kk.push_back(b/a);
	kk.push_back(b/a);
	for(int i=2; i<loaded_samples; i++)
	{
		hh.push_back( (d[i] - c * hh[i-1])/(a - c*kk[i-1]) );
		kk.push_back( b/(a - c * kk[i-1]) );
	}

	float *mm;
	checkCudaErrors(cudaMallocManaged(&mm, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	mm[hh.size()-1] = hh[hh.size()-1];
	for(size_t i=hh.size()-2; i > 0; i--){
		mm[i] = hh[i] - kk[i] * mm[i+1];
	}
	mm[0] = 0;
	mm[loaded_samples-1] = 0;

	kernel2<<<ceil((float)loaded_samples/thread_N), thread_N, 0, stream>>>(*dst, src, mm, samplerate, loaded_samples);
	cudaStreamSynchronize(stream);

	cudaFree(d);
	cudaFree(mm);
}


template<typename T>
void WaveData::LoadData(std::ifstream &fin, int channel)
{
	float b = (std::numeric_limits<T>::min() < 0)? 0: std::numeric_limits<T>::max() / 2.0;
	float a = (std::numeric_limits<T>::min() < 0)? std::numeric_limits<T>::max(): std::numeric_limits<T>::max() / 2.0;
	const float scale = 0.5;
	T ldata = b;
	T rdata = b;
	long count = 0;
	long readcount = header.dataChunk.chunkSize / channel / sizeof(T);
	readcount++;
	readcount-=start_sample;
	if(length_sample > 0)
		readcount = length_sample;

	if(channel == 1){
		checkCudaErrors(cudaMallocManaged(&left, readcount*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(left, readcount*sizeof(float), cudaMemAdviseSetReadMostly, 0));
		memset(left, 0, readcount*sizeof(float));
		mono = left;
	}
	else if(channel == 2){
		checkCudaErrors(cudaMallocManaged(&mono, readcount*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMallocManaged(&left, readcount*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMallocManaged(&right, readcount*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(mono, readcount*sizeof(float), cudaMemAdviseSetReadMostly, 0));
		checkCudaErrors(cudaMemAdvise(left, readcount*sizeof(float), cudaMemAdviseSetReadMostly, 0));
		checkCudaErrors(cudaMemAdvise(right, readcount*sizeof(float), cudaMemAdviseSetReadMostly, 0));
		memset(mono, 0, readcount*sizeof(float));
		memset(left, 0, readcount*sizeof(float));
		memset(right, 0, readcount*sizeof(float));
	}
	
	float *lp = left;
	float *rp = right;
	float *mp = mono;

	while(start_sample < 0){
		count++;
		if(channel == 1){
			*lp++ = (ldata - b)/a*scale;
		}
		else if(channel == 2){
			*lp++ = (ldata - b)/a*scale;
			*rp++ = (rdata - b)/a*scale;
			*mp++ = ((ldata - b)/a*scale + (rdata - b)/a*scale)/2;
		}
		start_sample++;
	}
	while(start_sample < 0){
		if(channel == 1){
			fin.read((char *)&ldata, sizeof(T));
		}
		else if(channel == 2){
			fin.read((char *)&ldata, sizeof(T));
			fin.read((char *)&rdata, sizeof(T));
		}
		start_sample--;
	}

	while(!fin.eof() && count <= readcount){
		count++;
		if(channel == 1){
			*lp++ = (ldata - b)/a*scale;

			fin.read((char *)&ldata, sizeof(T));
		}
		else if(channel == 2){
			*lp++ = (ldata - b)/a*scale;
			*rp++ = (rdata - b)/a*scale;
			*mp++ = ((ldata - b)/a*scale + (rdata - b)/a*scale)/2;

			fin.read((char *)&ldata, sizeof(T));
			fin.read((char *)&rdata, sizeof(T));
		}
	}

	printf("%ld samples readed\n", count);
	loaded_samples = count;
	samplerate = header.fmtChunk.samplesPerSec;

	printf("InterpolateSpline\n");
	if(channel == 1){
		cudaStream_t stream1;
		cudaStreamCreate(&stream1);
		InterpolateSpline(&leftp, left, stream1);
		monop = leftp;
		cudaStreamDestroy(stream1);
	}
	else if (channel == 2){
		cudaStream_t stream1;
		cudaStreamCreate(&stream1);
		InterpolateSpline(&leftp, left, stream1);
		InterpolateSpline(&rightp, right, stream1);
		InterpolateSpline(&monop, mono, stream1);
		cudaStreamDestroy(stream1);
	}
}

WaveData::WaveData(char *filename, double start, double length)
	: left(NULL), right(NULL), mono(NULL),
	leftp(NULL), rightp(NULL), monop(NULL)
{
	loaded_samples = -1;
	std::ifstream fin(filename, std::ios::binary);
	if(fin){
		fin.read((char *)&header.riffChunk, sizeof(RIFF_CHUNK));
		fin.read((char *)&header.fmtChunk, sizeof(FMT_CHUNK));
		fin.seekg(header.fmtChunk.chunkSize + 8 - sizeof(FMT_CHUNK),std::ios_base::cur);
		fin.read((char *)&header.dataChunk, sizeof(DATA_CHUNK));
		if(strncmp(header.riffChunk.chunkID,"RIFF", 4) != 0){
			return;
		}
		if(strncmp(header.riffChunk.chunkFormType,"WAVE", 4) != 0){
			return;
		}
		if(strncmp(header.fmtChunk.chunkID,"fmt ", 4) != 0){
			return;
		}
		if(strncmp(header.dataChunk.chunkID,"data", 4) != 0){
			return;
		}
		if(header.fmtChunk.waveFormatType != 1){
			fprintf(stderr, "not PCM WAVE file\n");
			return;
		}
		header_endp = fin.tellg();
		printf("%ld bytes header end\n", header_endp);

		start_sample = start * header.fmtChunk.samplesPerSec;
		length_sample = length * header.fmtChunk.samplesPerSec;

		if(header.fmtChunk.bitsPerSample == 8){
			LoadData<uint8_t>(fin, header.fmtChunk.formatChannel);
		}
		else if(header.fmtChunk.bitsPerSample == 16){
			LoadData<int16_t>(fin, header.fmtChunk.formatChannel);
		}
	}
}

WaveData::WaveData(char *filename)
	: WaveData(filename, 0, -1)
{
}


WaveData::WaveData(const WaveData& other)
	: left(NULL), right(NULL), mono(NULL),
	leftp(NULL), rightp(NULL), monop(NULL)
{
	printf("WaveData Copy\n");
	header = {0};
	header_endp = 0;
	start_sample = 0;
	length_sample = 0;
	loaded_samples = other.loaded_samples;
	samplerate = other.samplerate;	
	if(other.left){
		checkCudaErrors(cudaMallocManaged(&left, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(left, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
		memcpy(left, other.left, loaded_samples*sizeof(float));
	}
	if(other.right){
		checkCudaErrors(cudaMallocManaged(&right, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(right, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
		memcpy(right, other.right, loaded_samples*sizeof(float));
	}
	if(other.mono){
		checkCudaErrors(cudaMallocManaged(&mono, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(mono, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
		memcpy(mono, other.mono, loaded_samples*sizeof(float));
	}
	printf("InterpolateSpline\n");
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	if(left)
		InterpolateSpline(&leftp, left, stream1);
	if(right)
		InterpolateSpline(&rightp, right, stream1);
	if(mono)
		InterpolateSpline(&monop, mono, stream1);
	cudaStreamDestroy(stream1);
}

WaveData& WaveData::operator=(const WaveData& other)
{
	if(this != &other){
		printf("WaveData Operator=\n");
		header = {0};
		header_endp = 0;
		start_sample = 0;
		length_sample = 0;
		if(left != mono)
			cudaFree(mono);
		if(leftp != mono)
			cudaFree(monop);
		if(left != right)
			cudaFree(right);
		if(leftp != rightp)
			cudaFree(rightp);
		cudaFree(left);
		cudaFree(leftp);
		left = leftp = NULL;
		right = rightp = NULL;
		mono = monop = NULL;
		loaded_samples = other.loaded_samples;
		samplerate = other.samplerate;	
		if(other.left){
			checkCudaErrors(cudaMallocManaged(&left, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
			checkCudaErrors(cudaMemAdvise(left, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
			memcpy(left, other.left, loaded_samples*sizeof(float));
		}
		if(other.right){
			checkCudaErrors(cudaMallocManaged(&right, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
			checkCudaErrors(cudaMemAdvise(right, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
			memcpy(right, other.right, loaded_samples*sizeof(float));
		}
		if(other.mono){
			checkCudaErrors(cudaMallocManaged(&mono, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
			checkCudaErrors(cudaMemAdvise(mono, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
			memcpy(mono, other.mono, loaded_samples*sizeof(float));
		}
		printf("InterpolateSpline\n");
		cudaStream_t stream1;
		cudaStreamCreate(&stream1);
		if(left)
			InterpolateSpline(&leftp, left, stream1);
		if(right)
			InterpolateSpline(&rightp, right, stream1);
		if(mono)
			InterpolateSpline(&monop, mono, stream1);
		cudaStreamDestroy(stream1);
	}
	return *this;
}

void WaveData::LoadFrom(const std::vector<float> &smono)
{
	printf("WaveData LoadFrom\n");
	size_t len = loaded_samples;
	if(len > smono.size()) len = smono.size();

	if(!mono){
		checkCudaErrors(cudaMallocManaged(&mono, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(mono, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	}
	memcpy(mono, smono.data(), len*sizeof(float));
	if(len < loaded_samples)
		memset(&mono[len], 0, (loaded_samples - len)*sizeof(float));

	printf("InterpolateSpline\n");
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	InterpolateSpline(&monop, mono, stream1);
	if(left && mono != left){
		memcpy(left, mono, loaded_samples*sizeof(float));
		InterpolateSpline(&leftp, left, stream1);
	}
	if(right && mono != right){
		memcpy(right, mono, loaded_samples*sizeof(float));
		InterpolateSpline(&rightp, right, stream1);
	}
	cudaStreamDestroy(stream1);
}

void WaveData::LoadFrom(const std::vector<float> &sleft, const std::vector<float> &sright)
{
	printf("WaveData LoadFromStereo\n");
	size_t len = loaded_samples;
	if(len > sleft.size()) len = sleft.size();
	if(len > sright.size()) len = sright.size();

	printf("InterpolateSpline\n");
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	
	if(!left){
		checkCudaErrors(cudaMallocManaged(&left, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(left, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	}
	memcpy(left, sleft.data(), len*sizeof(float));
	if(len < loaded_samples)
		memset(&left[len], 0, (loaded_samples - len)*sizeof(float));
	InterpolateSpline(&leftp, left, stream1);
	
	if(!right){
		checkCudaErrors(cudaMallocManaged(&right, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(right, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	}
	memcpy(right, sright.data(), len*sizeof(float));
	if(len < loaded_samples)
		memset(&right[len], 0, (loaded_samples - len)*sizeof(float));
	InterpolateSpline(&rightp, right, stream1);
	
	if(!mono){
		checkCudaErrors(cudaMallocManaged(&mono, loaded_samples*sizeof(float), CU_MEM_ATTACH_GLOBAL));
		checkCudaErrors(cudaMemAdvise(mono, loaded_samples*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	}
	if(left != mono){
		memset(mono, 0, loaded_samples*sizeof(float));
		for(size_t i = 0; i < len; i++){
			mono[i] = (left[i] + right[i]) / 2;
		}
		InterpolateSpline(&monop, mono, stream1);
	}
	cudaStreamDestroy(stream1);
}

WaveData::~WaveData()
{
	printf("WaveData Destructor\n");
	if(left != mono)
		cudaFree(mono);
	if(leftp != mono)
		cudaFree(monop);
	if(left != right)
		cudaFree(right);
	if(leftp != rightp)
		cudaFree(rightp);
	cudaFree(left);
	cudaFree(leftp);
}

