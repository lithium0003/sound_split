#include "findphase.hpp"
#include <helper_cuda.h>

#define thread_N 32

__global__ void kernel_phase(const float *amp, const float *org, int samplerate, int srclen, int nphase, int nfreq, const float *phase, const float *freq, float *result)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int fidx = idx / nphase;
	int pidx = idx % nphase;
	float fq = freq[fidx];
	float ph = phase[pidx];

	if(fidx >= nfreq){
		printf("idx error in kernel_phase\n");
		return;
	}

	float ret = 0;
	for(int i = 0; i < srclen; i++){
		float v = amp[i] * sinf( 2.0 * M_PI * i * fq / samplerate + ph);
		ret += fabs(org[i] - v);
	}
	result[idx] = ret;
}

std::vector<float> FindMatchWave(int samplefq, const std::vector<float> &ampwav, const std::vector<float> &cfreq, const float *orgwave)
{
	printf("FindMatchWave\n");

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	float *amp, *org;
	int len = ampwav.size();
	std::vector<float> ret(len);

	checkCudaErrors(cudaMallocManaged(&amp, len*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(amp, len*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	checkCudaErrors(cudaMallocManaged(&org, len*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(org, len*sizeof(float), cudaMemAdviseSetReadMostly, 0));

	memcpy(amp, ampwav.data(), len*sizeof(float));
	memcpy(org, orgwave, len*sizeof(float));

	float *result;
	float *phase, *freq;
	int nphase = thread_N * 128;
	int nfreq = thread_N * 128;
	checkCudaErrors(cudaMallocManaged(&phase, nphase*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(phase, nphase*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	checkCudaErrors(cudaMallocManaged(&freq, nfreq*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(freq, nfreq*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	
	checkCudaErrors(cudaMallocManaged(&result, nphase*nfreq*sizeof(float), CU_MEM_ATTACH_GLOBAL));

	for(int i = 0; i < nphase; i++){
		phase[i] = 2.0 * M_PI * i / nphase;
	}

	std::vector<int> retidx;
	for(int i = 0; i < nfreq*nphase; i++){
		retidx.push_back(i);
	}

	int remlen = len;
	float *retp = ret.data();
	const float *freqp = cfreq.data();

	const int analyse = 100;
	const int overlap = 10;
	float *ap = amp;
	float *op = org;
	while(remlen > 0){
		float cf = *freqp;

		int wavstep = analyse;
		if(remlen < wavstep)
			wavstep = remlen;

		for(int i = -nfreq/2; i < nfreq/2 ; i++){
			freq[i+nfreq/2] = cf + 0.01 * cf * 2 / nfreq * i;
		}

		kernel_phase<<<nphase*nfreq/thread_N, thread_N, 0, stream>>>(ap, op, samplefq, remlen, nphase, nfreq, phase, freq, result);
		cudaStreamSynchronize(stream);

		std::sort(retidx.begin(), retidx.end(), 
				[result](const int a, const int b){
					return result[a] < result[b];
				});


		float minphase = phase[retidx[0] % nphase];
		float minfreq = freq[retidx[0] / nphase];
		printf("freq %f, phase %f = %f\n", minfreq, minphase*180/M_PI, result[retidx[0]]);

		for(int i = 0; i < std::min(wavstep, analyse/overlap); i++){
			float ph = 2.0 * M_PI * minfreq * i / samplefq + minphase;
			retp[i] = ap[i] * sinf(ph);
		}

		retp += analyse/overlap;
		ap += analyse/overlap;
		op += analyse/overlap;
		remlen -= analyse/overlap;
		freqp += analyse/overlap;
	}

	cudaStreamDestroy(stream);

	cudaFree(amp);
	cudaFree(org);
	cudaFree(phase);
	cudaFree(freq);
	cudaFree(result);

	return ret;
}

float FindPhase(float &cfreq, int samplefq, const std::vector<float> &ampwav, const float *orgwave)
{
	//printf("FindPhase %f Hz\n", cfreq);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	float *amp, *org;
	int len = ampwav.size();

	checkCudaErrors(cudaMallocManaged(&amp, len*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(amp, len*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	checkCudaErrors(cudaMallocManaged(&org, len*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(org, len*sizeof(float), cudaMemAdviseSetReadMostly, 0));

	memcpy(amp, ampwav.data(), len*sizeof(float));
	memcpy(org, orgwave, len*sizeof(float));


	float *result;
	float *phase, *freq;
	int nphase = thread_N * 256;
	int nfreq = thread_N * 8;
	checkCudaErrors(cudaMallocManaged(&phase, nphase*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(phase, nphase*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	checkCudaErrors(cudaMallocManaged(&freq, nfreq*sizeof(float), CU_MEM_ATTACH_GLOBAL));
	checkCudaErrors(cudaMemAdvise(freq, nfreq*sizeof(float), cudaMemAdviseSetReadMostly, 0));
	
	checkCudaErrors(cudaMallocManaged(&result, nphase*nfreq*sizeof(float), CU_MEM_ATTACH_GLOBAL));

	float dphase1 = 2.0 * M_PI / nphase;
	float dfreq1 = 0.01 * cfreq / nfreq;
	for(int i = 0; i < nphase; i++){
		phase[i] = 2.0 * M_PI * i / nphase;
	}
	for(int i = 0; i < nfreq ; i++){
		freq[i] = cfreq + 0.01 * cfreq / nfreq * (i - nfreq/2);
	}


	kernel_phase<<<nphase*nfreq/thread_N, thread_N, 0, stream>>>(amp, org, samplefq, len, nphase, nfreq, phase, freq, result);

	std::vector<int> retidx;
	for(int i = 0; i < nfreq*nphase; i++){
		retidx.push_back(i);
	}
	cudaStreamSynchronize(stream);

	std::sort(retidx.begin(), retidx.end(), 
			[result](const int a, const int b){
				return result[a] < result[b];
			});

	float minphase = phase[retidx[0] % nphase];
	float minfreq = freq[retidx[0] / nphase];
	float value1 = result[retidx[0]];

	//printf("queue %f freq %f(%f), phase %f(%f) = %f\n", 
	//		cfreq, minfreq, dfreq1, minphase*180/M_PI, dphase1*180/M_PI, value1); 
	
	cudaStreamDestroy(stream);

	cudaFree(amp);
	cudaFree(org);
	cudaFree(phase);
	cudaFree(freq);
	cudaFree(result);

	cfreq = minfreq;
	return minphase;
}
