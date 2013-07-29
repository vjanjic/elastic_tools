/**
 * ChunkingKernel.cu
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef CHUNKINGKERNEL_CU_
#define CHUNKINGKERNEL_CU_

#include "rabin_fingerprint/Chunker.h"
#include  "cuda_runtime.h"
#include "ResourceManagement.h"
#include "KernelStarter_CS.h"
#include "BitFieldArray.h"
#include <iostream>
#include <fstream>      // std::ifstream
#include "../../misc/Macros.h"

#include "openssl/sha.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ int getThrID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ void getThreadBounds(threadBounds* bounds, int dataLn, int threadsUsed, int thrID, int workPerThr) {

	bounds->start = thrID * workPerThr;

	//ACCOUTN FOR ANY LEFTOVER DATA THAT CANNOT BE DISTRIBUTED ;)
	bounds->end = (thrID == threadsUsed - 1) ? bounds->end = dataLn : bounds->start + workPerThr;

}

__global__ void findBreakPointsFreeMode(rabinData* deviceRabin, BYTE* data, int dataLen, bitFieldArray results, int threadsUsed, int workPerThread,
		int divisor) {

	int thrID = getThrID();


	if (thrID < threadsUsed) {

		threadBounds dataBounds;

		getThreadBounds(&dataBounds, dataLen, threadsUsed, thrID, workPerThread);


		chunkDataFreeMode(deviceRabin, data, dataBounds, divisor, results, threadsUsed);
	}
}

void startCreateBreakpointsKernel(int blocksSize, int numBlocks, rabinData* deviceRabin, BYTE* deviceData, int dataLen, bitFieldArray results,
		int threadsUsed, int workPerThread, int D, cudaStream_t stream) {
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);


	findBreakPointsFreeMode<<<numBlocks, blocksSize,0,stream>>>(deviceRabin, deviceData, dataLen, results, threadsUsed, workPerThread, D);

	gpuErrchk(cudaGetLastError());

	//cudaThreadSynchronize();
}

int __host__ getSizeOfBPArray(int dataLn, int minThreshold) {
	return (dataLn % minThreshold == 0) ? dataLn / minThreshold : (dataLn / minThreshold) + 1;
}

 cudaFuncAttributes getChunkingKernelProperties() {
	cudaFuncAttributes attributes;
	cudaFuncGetAttributes(&attributes, findBreakPointsFreeMode);
	return attributes;
}

#endif /* CHUNKINGKERNEL_CU_ */
