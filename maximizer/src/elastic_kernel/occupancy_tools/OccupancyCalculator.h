/**
 * OccupancyCalculator.hpp
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
inline  size_t getMaxResidentBlocksPerSM(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	size_t hardwareLimit = getHardwareLimit(deviceProps, blockSize);
	size_t sMemLimit = getSharedMemLimit(deviceProps, kernelProps);
	size_t registerLimit = getRegisterLimit(deviceProps, kernelProps, blockSize);

	return min3(hardwareLimit, sMemLimit, registerLimit);
}

inline size_t getNumRegistersPerBlock(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	size_t numerWarpsNeeded = (blockSize + (deviceProps.warpSize - 1)) / deviceProps.warpSize; // we need to devide and round UP to warpsize
	numerWarpsNeeded = ceilTo(numerWarpsNeeded, getWarpAllocationGranularity(deviceProps)); // again CEIL up to the war allocation granularity, depending on architecture;

	return kernelProps.numRegs * deviceProps.warpSize * numerWarpsNeeded;
}

inline BlockUsage getBlockUsageStats(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	BlockUsage usage;
	usage.numThreads = blockSize;
	usage.blocksPerSM = getMaxResidentBlocksPerSM(deviceProps, kernelProps, blockSize);
	usage.numRegisters = getNumRegistersPerBlock(deviceProps, kernelProps, blockSize);
	usage.sharedMemory = getSharedMemNeeded(kernelProps, deviceProps);

	return usage;
}

inline void reduceBlockToFit(size_t usage, size_t limit, PhysicalConfiguration &configuration) {
	size_t currentUsage = configuration.blocksPerGrid * usage;

	if (currentUsage > limit) {

		int deficit = currentUsage - limit;
		int decrement = (deficit / usage);
		std::cout << "Decrement : "<< decrement << std::endl;

		configuration.blocksPerGrid = configuration.blocksPerGrid - decrement;
	}
}

inline PhysicalConfiguration limitUsage(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, blockParams_logical blkL, gridParams_logical grdL,
		KernelLimits limits) {

	//calcualte blocksize and number of blocks per grid
	size_t blockSize = blkL.blkDim_X * blkL.blkDim_Y * blkL.blkDim_Z;
	size_t blocks = grdL.grdDim_X * grdL.grdDim_Y;

	std::cout<< blockSize << " " << blocks << std::endl;

	BlockUsage usage = getBlockUsageStats(deviceProps, kernelProps, blockSize);

	printUsage(usage);

	size_t maximumResidentBLocks = usage.blocksPerSM * deviceProps.multiProcessorCount;
	size_t blocksFinal = blocks;
	size_t threadsFinal = blockSize;

	blocksFinal = min3(blocks, maximumResidentBLocks, limits.blocks);

	//change threads
/*	int incrementThreads = ((blocks * blockSize) - (blocksFinal * blockSize)) / blocksFinal;
	threadsFinal = threadsFinal + incrementThreads;*/

	PhysicalConfiguration result;
	result.threadsPerBlock = threadsFinal;
	result.blocksPerGrid = blocksFinal;

	printHysicalConfig(result);
	reduceBlockToFit(usage.sharedMemory, limits.sharedMem, result);
	std::cout << "Thread usage: " << usage.numThreads << " Limit on threads: " << limits.threads << std::endl;
	reduceBlockToFit(usage.numThreads, limits.threads, result);
	reduceBlockToFit(usage.numRegisters, limits.registers, result);
	printHysicalConfig(result);

	return result;
}

#endif /* OCCUPANCYCALCULATOR_HPP_ */
