/**
 * OccupancyCalculator.hpp
 *
 *
 * This file contains functions, which purpose is to determine maximum occupancy
 * for particular kernels on particular GPU configuration. Most of this involves
 * calculations that are described in the NVIDIA hardware implementation guide.
 *
 *
 *
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYCALCULATOR_HPP_
#define OCCUPANCYCALCULATOR_HPP_
#include "OccupancyData.h"
#include "OccupancyLimits.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "../AbstractElasticKernel.hpp"
#include <boost/shared_ptr.hpp>

struct OccupancyInformation {
	int optimalThreadBlockSize;
	double respectiveSMOccupancy;
};

/**
 *
 * Function to obtain the hardware configuration information for the device
 *
 * @return cudaDeviceProp
 */
inline cudaDeviceProp getGPUProperties() {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	return props;
}

/**
 * This function calculates the maximum resident blocks per SM for a particular kernel
 * The function takes into account the limitations imposed by register pressure,
 * shared memory and raw hardware specifications
 *
 * @param deviceProps the properties of the device
 * @param kernelProps the properties of the kernel
 * @param blockSize the block size of the kernel
 *
 * @return maximum active blocks per SM
 */
inline size_t getMaxResidentBlocksPerSM(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	size_t hardwareLimit = getHardwareLimit(deviceProps, blockSize);
	size_t sMemLimit = getSharedMemLimit(deviceProps, kernelProps);
	size_t registerLimit = getRegisterLimit(deviceProps, kernelProps, blockSize);

	//std::cout << hardwareLimit << " " << sMemLimit << " " << registerLimit << std::endl;

	return min3(hardwareLimit, sMemLimit, registerLimit);
}

/**
 * This function retrieves the number of registers that are needed for a block
 *
 * @param deviceProps the device properties
 * @param kernelProps the kernel properties
 * @param blockSize the size of the block
 * @return the number of registers needed per block
 */
inline size_t getNumRegistersPerBlock(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	size_t numerWarpsNeeded = (blockSize + (deviceProps.warpSize - 1)) / deviceProps.warpSize; // we need to devide and round UP to warpsize
	numerWarpsNeeded = ceilTo(numerWarpsNeeded, getWarpAllocationGranularity(deviceProps)); // again CEIL up to the war allocation granularity, depending on architecture;

	return kernelProps.numRegs * deviceProps.warpSize * numerWarpsNeeded;
}

/**
 * This function compiles a block usage data structure for a particular kernel. The information returned
 * includes the amount of threads requested per block, the shared memory and the registers.
 *
 * @param deviceProps the properties of the device
 * @param kernelProps the kernel properties
 * @param blockSize the blck size for the kernel
 * @return
 */
inline BlockUsage getBlockUsageStats(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	BlockUsage usage;
	size_t numThreads = blockSize;
	size_t blocksPerSM = getMaxResidentBlocksPerSM(deviceProps, kernelProps, blockSize);
	size_t numRegisters = getNumRegistersPerBlock(deviceProps, kernelProps, blockSize);
	size_t sharedMemory = getSharedMemNeeded(kernelProps, deviceProps);

	return BlockUsage(sharedMemory, numThreads, numRegisters, blocksPerSM);
}

/**
 * Given a limits value and a usage value, this function molds the execution parameters
 * of the block in order to fit it into the limit.
 *
 * @param usagePerBlock the usage value
 * @param limitPerGPU the limit value
 * @param params reference to the launch paramters of the device
 */
inline void reduceBlocksToFitOnGPU(size_t usagePerBlock, size_t limitPerGPU, LaunchParameters &params) {
	// calcualte how much is the usage for the whole card...

	size_t currentUsage = params.getBlocksPerGrid() * usagePerBlock;

	if (currentUsage > limitPerGPU) {
		// if we exceed the limit.. we need to decrement

		int deficit = currentUsage - limitPerGPU; //  calculatye the total deficit

		size_t decrement = ceil((double) deficit / usagePerBlock); // calcualte how many blocks we need to trim in order to fit

		size_t decreasedBlocks = params.getBlocksPerGrid() - decrement;

		params.setBlocks(decreasedBlocks);
	}
}

/**
 *
 * Given kernel properties, device properties , launch parameters and kernel limits,
 * this function molds the configuration of the kernel in order to fit it into the
 * limits provided. Those limits are shared memory, registers , total number of threads
 *
 * @param deviceProps the device properties
 * @param kernelProps the kernel properties
 * @param lParams the launch parameters for the kernel
 * @param limits the limits on the kernel
 *
 * @return the molded launch parameters, which ensure that the kernel fits into the limits imposed
 */
inline LaunchParameters limitUsage(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, LaunchParameters lParams, KernelLimits limits) {
	// we get the occupancy information for the particular block
	BlockUsage usage = getBlockUsageStats(deviceProps, kernelProps, lParams.getThreadsPerBlock());
	// we calculate the maximum number of resident blocks of this size on the GPU
	size_t maximumResidentBLocks = usage.getNumBlocksPerSM() * deviceProps.multiProcessorCount;

	// constructing physical configuration... based on calculated number of blocks :)
	size_t blocksPhysical = min3(lParams.getBlocksPerGrid(), maximumResidentBLocks, limits.getNumBlocks());
	size_t threadsPhysical = lParams.getThreadsPerBlock();
	LaunchParameters result = LaunchParameters(threadsPhysical, blocksPhysical);

	// now we need to further limit this physical  configuration, which of course is pain

	reduceBlocksToFitOnGPU(usage.getSharedMem(), limits.getSharedMem(), result);
	reduceBlocksToFitOnGPU(usage.getNumThreads(), limits.getNumThreads(), result);
	reduceBlocksToFitOnGPU(usage.getNumRegisters(), limits.getNumRegisters(), result);

	return result;
}

/**
 * Given a pointer to an elastic kernel and a set of limits, the kernel's configuration is
 * molded in order to fit into the hardware limits
 *
 * @param kernel a shared pointer to the kernel
 * @param limits a set of device limits
 *
 * @return the new launch parameters for the kernel
 */
inline LaunchParameters limitKernel(boost::shared_ptr<AbstractElasticKernel> kernel, KernelLimits limits) {
	LaunchParameters params = kernel.get()->getLaunchParams();
	cudaFuncAttributes attrs = kernel.get()->getKernelProperties();
	cudaDeviceProp props = getGPUProperties();
	return limitUsage(props, attrs, params, limits);

}

/**
 * This function returns the theoretical compute occupancy of a particular kernel using
 * heuristics that take into account the characteristics of the particular kernel as well
 * as the device properties
 *
 * @param kernel
 * @return
 */
inline double getOccupancyForKernel(boost::shared_ptr<AbstractElasticKernel> kernel) {
	cudaDeviceProp gpuConfiguration = getGPUProperties();
	size_t max_occupancy = gpuConfiguration.maxThreadsPerMultiProcessor;
	size_t threadNum = min3(kernel.get()->getKernelProperties().maxThreadsPerBlock, gpuConfiguration.maxThreadsPerMultiProcessor,
			kernel.get()->getLaunchParams().getThreadsPerBlock());

	size_t maxBlocksPerSm = getMaxResidentBlocksPerSM(gpuConfiguration, kernel.get()->getKernelProperties(), threadNum);
	size_t occupancy = threadNum * maxBlocksPerSm;

	return (double) occupancy / max_occupancy;

}

/**
 * Given a kernel, this function retrieves its memory occupancy. This value is the fraction
 * of memory that the kernel requests from the global memory of the device.
 *
 * @param kernel a pointer to a kernel
 *
 * @return the memory occupancy (global memory)
 */
inline double getMemoryOccupancyForKernel(boost::shared_ptr<AbstractElasticKernel> kernel) {
	cudaDeviceProp deviceProps = getGPUConfiguration();
	double totalGPUmem = (double) deviceProps.totalGlobalMem;
	double result = (double) kernel.get()->getMemoryConsumption() / totalGPUmem;

	return result;
}

/**
 *
 * This function applies a brute force iterative approach to finding the optimal threadblock size
 * for a particular kernel based on the kernel characteristics as well as the device hardware
 * capabilities.
 *
 * @param kernel a pointer to the kernel.
 * @return
 */
inline size_t getOptimalBlockSize(boost::shared_ptr<AbstractElasticKernel> kernel) {

	cudaDeviceProp gpuConfiguration = getGPUProperties();
	size_t max_occupancy = gpuConfiguration.maxThreadsPerMultiProcessor;
	size_t largestThrNum = min2(kernel.get()->getKernelProperties().maxThreadsPerBlock, gpuConfiguration.maxThreadsPerMultiProcessor);
	size_t threadGranularity = gpuConfiguration.warpSize;

	size_t maxBLockSize = 0;
	size_t highestOcc = 0;

	for (size_t blocksize = largestThrNum; blocksize != 0; blocksize -= threadGranularity) {

		size_t maxBlocksPerSm = getMaxResidentBlocksPerSM(gpuConfiguration, kernel.get()->getKernelProperties(), blocksize);
		size_t occupancy = blocksize * maxBlocksPerSm;

		//std::cout << "Trying blocksize "  << blocksize << " " << occupancy  << " " << highestOcc  <<  " " << (double)occupancy / max_occupancy << std::endl;

		if (occupancy > highestOcc) {
			maxBLockSize = blocksize;
			highestOcc = occupancy;
		}

		// early out, can't do better
		if (highestOcc == max_occupancy)
			break;
	}
	//printf("Blocksize %d\n", highestOcc);

	return maxBLockSize;

}

#endif /* OCCUPANCYCALCULATOR_HPP_ */
