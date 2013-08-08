/**
 * VectorAdditionKernel.cuh
 *
 *  Created on: Jul 29, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef VECTORADDITIONKERNEL_CUH_
#define VECTORADDITIONKERNEL_CUH_

#include <stdio.h>
#include "Kernel_Starter_VA.h"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void addVectors(int *A, int *B, int numElements, int workPerThread, int totalThreads) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int from = id * workPerThread;
	int till = (id == totalThreads - 1) ? numElements : from + workPerThread;
	for (; from < till; ++from) {
		int sum = A[from] + B[from];

		B[from] = sum;

	}

}

void startAddVectorsKernel(size_t threads, size_t blocks, int *A, int *B, int numElements, cudaStream_t stream) {
	int totalThreads = threads * blocks;

	int workPerThread = numElements / totalThreads;
	addVectors<<<blocks, threads, 0, stream>>>(A, B, numElements, workPerThread, totalThreads);

}

cudaFuncAttributes getVectorAddKernelProperties() {
	cudaFuncAttributes attributes;
	cudaFuncGetAttributes(&attributes, addVectors);
	return attributes;
}

#endif /* VECTORADDITIONKERNEL_CUH_ */
