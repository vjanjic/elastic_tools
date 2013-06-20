/**
 * OccupancyCalculator.hpp
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYCALCULATOR_HPP_
#define OCCUPANCYCALCULATOR_HPP_
#include "OccupancyData.hpp"
#include "OccupancyLimits.hpp"

size_t getMaxResidentBlocksPerSM(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	size_t hardwareLimit = getHardwareLimit(deviceProps,blockSize);
	size_t sMemLimit = getSharedMemLimit(deviceProps, kernelProps);
	size_t registerLimit = getRegisterLimit(deviceProps, kernelProps, blockSize);

	return min3(hardwareLimit, sMemLimit, registerLimit);
}

size_t getNumRegistersPerBlock(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {


	size_t numerWarpsNeeded = (blockSize + (deviceProps.warpSize - 1)) / deviceProps.warpSize; // we need to devide and round UP to warpsize
	numerWarpsNeeded = ceilTo(numerWarpsNeeded,getWarpAllocationGranularity(deviceProps) ); // again CEIL up to the war allocation granularity, depending on architecture;

	return kernelProps.numRegs * deviceProps.warpSize * numerWarpsNeeded;
}

BlockUsage getBlockUsageStats(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {

	BlockUsage usage;
	usage.numThreads = blockSize;
	usage.blocksPerSM = getMaxResidentBlocksPerSM(deviceProps, kernelProps, blockSize);
	usage.numRegisters = getNumRegistersPerBlock(deviceProps, kernelProps, blockSize);
	usage.sharedMemory = getSharedMemNeeded(kernelProps, deviceProps);

	return usage;
}

#endif /* OCCUPANCYCALCULATOR_HPP_ */
