/**
 * dummy_elastic.cu
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "declarations.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "cuda.h"
#include <iostream>
extern "C" PhysicalConfiguration scale_dummy_elastic(blockParams_logical blk_logical, gridParams_logical grd_logical, KernelLimits limits);
extern "C" void lauch_dummy_elastic(blockParams_logical blk_logical, gridParams_logical grd_logical, KernelLimits limits);

__global__ void dummy_elastic(clock_t *d_o, clock_t clock_count) {

	int thrID = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Thread %d\n", thrID);

	unsigned int start_clock = (unsigned int) clock();

	clock_t clock_offset = 0;

	while (clock_offset < clock_count) {
		unsigned int end_clock = (unsigned int) clock();

		// The code below should work like
		// this (thanks to modular arithmetics):
		//
		// clock_offset = (clock_t) (end_clock > start_clock ?
		//                           end_clock - start_clock :
		//                           end_clock + (0xffffffffu - start_clock));
		//
		// Indeed, let m = 2^32 then
		// end - start = end + m - start (mod m).

		clock_offset = (clock_t) (end_clock - start_clock);
	}

	d_o[0] = clock_offset;

}

PhysicalConfiguration scale_dummy_elastic(blockParams_logical blk_logical, gridParams_logical grd_logical, KernelLimits limits) {

	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, 0);

	cudaFuncAttributes attributes;
	cudaFuncGetAttributes(&attributes, dummy_elastic);
	PhysicalConfiguration result = limitUsage(deviceProperties, attributes, blk_logical, grd_logical, limits);

	return result;
}

void lauch_dummy_elastic(blockParams_logical blk_logical, gridParams_logical grd_logical, KernelLimits limits) {
	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, 0);
	PhysicalConfiguration lauchConfig = scale_dummy_elastic(blk_logical, grd_logical, limits);

	float kernel_time = 10; // time the kernel should run in ms

	//clock_t *a = 0;                     // pointer to the array data in host memory
	//checkCudaErrors(cudaMallocHost((void **)&a, sizeof(clock_t)));

	// allocate device memory
	clock_t *d_a = 0;             // pointers to data and init value in the device memory
	cudaMalloc((void **) &d_a, sizeof(clock_t));

	clock_t time_clocks = (clock_t) (kernel_time * deviceProperties.clockRate);
	//std::cout << lauchConfig.blocksPerGrid << std::endl;
	//std::cout << lauchConfig.threadsPerBlock << std::endl;

	//dummy_elastic<<<10, 10>>>(d_a, time_clocks);

	cudaFree(d_a);
}
int main() {

	gridParams_logical grid = getLogicalGrid(10, 1);
	blockParams_logical block = getLogicalBlock(320, 1, 1);


	KernelLimits limits;
	limits.blocks = 3;
	limits.registers = 16000;
	limits.sharedMem = 100;
	limits.threads = 150;

	lauch_dummy_elastic(block,grid, limits);

	//
	//ElasticDummy dummyKernel(grid, block);
	//dummyKernel.runKernel();

	return 0;
}

