/**
 * OccupancyLimits.hpp
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYLIMITS_HPP_
#define OCCUPANCYLIMITS_HPP_
#include "OccupancyUtils.hpp"




size_t getRegisterLimit(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps, size_t blockSize) {
	/*
	 * Now we need to calculate the limits due to register pressure
	 */

	// we first get the register allocation granularity and the warp allocation one
	size_t registerAlocationGranularity = getRegisterAllocationGranularity(deviceProps);
	size_t warpAllocationGranularity = (deviceProps.major > 1) ? 1 : 2;

	/*
	 * We here calculate the number of warps needed for this number of threads per CTA.
	 * This is done by rounded devision to the number of threads per warp (which is the warp size)
	 * After that we check the compute capability.
	 */

	size_t numerWarpsNeeded = (blockSize + (deviceProps.warpSize - 1)) / deviceProps.warpSize; // we need to devide and round UP to warpsize
	/*
	 * here we CEIL up to 1 or 2, depending on the compute capability of the GPU
	 */
	numerWarpsNeeded = ceilTo(numerWarpsNeeded, warpAllocationGranularity); // again CEIL up to the war allocation granularity, depending on architecture;

	size_t maxBlocksRegisterLimit = 0;

	if (deviceProps.major <= 1) {
		/*
		 * We know that devices of compute capability of 1.x allocate registers per block.
		 * THerefore, number of registers per block would be the number of warps times
		 * times the num of registers per thread, all of that CEILED up to the register
		 * allocation size.
		 */
		size_t registersNeeded = kernelProps.numRegs * deviceProps.warpSize * numerWarpsNeeded;
		registersNeeded = ceilTo(registersNeeded, registerAlocationGranularity);
		if (registersNeeded > 0) {
			/*if the needed registers is more than 0  we get the limit of maximum blocks
			 *with respect to the register pressure, simply by dividing the avaible
			 *registers per block by the number of registers we need to see how many
			 *we can fit in.
			 *
			 */
			maxBlocksRegisterLimit = deviceProps.regsPerBlock / registersNeeded;
		} else {
			// if we do not need any registers, simply set the max to the max of blocks per SM
			maxBlocksRegisterLimit = getMaxSMBlocks(deviceProps);
		}
	} else {
		/*
		 * In case our device is of higher compute capability than one, we need to consider that
		 * registers are allocated per warps. So the number of registers per warp would be the number of
		 * registers per thread * number of threads per warp  CEILED_UP to the register
		 * allocation unit
		 */
		size_t registersPerWarp = ceilTo(kernelProps.numRegs * deviceProps.warpSize, registerAlocationGranularity);
		size_t numberOfWarpSchedulersPerSM = getNumWarpSchedulers(deviceProps); // depending on architecture (Kepler has a quad warp scheduler)
		size_t registersPerScheduler = deviceProps.regsPerBlock / numberOfWarpSchedulersPerSM;
		if (registersPerWarp > 0) {
			maxBlocksRegisterLimit = ((registersPerScheduler / registersPerWarp) * numberOfWarpSchedulersPerSM) / numerWarpsNeeded;
		} else {
			maxBlocksRegisterLimit = getMaxSMBlocks(deviceProps);
		}
	}

	return maxBlocksRegisterLimit;

}

size_t getSharedMemLimit(const cudaDeviceProp &deviceProps, const cudaFuncAttributes &kernelProps) {

	/*
	 * Now we need to consider how many blocks can we really have active due to the limit of our shared memory that is requested.
	 */


	size_t sharedMemoryNeeded = getSharedMemNeeded(kernelProps,deviceProps);

	size_t maxBlocksSMLimit = 0; // now we are ready to check out limit posed by shared memory
	if (sharedMemoryNeeded > 0) {
		maxBlocksSMLimit = deviceProps.sharedMemPerBlock / sharedMemoryNeeded;
	} else {
		// else we know we do not have any limit on that, so we set the max to be the maximum num of active blocks per SM
		maxBlocksSMLimit = getMaxSMBlocks(deviceProps);
	}

	return maxBlocksSMLimit;
}

size_t getHardwareLimit(const cudaDeviceProp &deviceProps, size_t blockSize) {
	/*
	 * First we start by examining the actual hardware limits
	 * of the GPU with respect to maximum active blocks of
	 * size N per SM....
	 */

	// we first need the maximum number of resident threads per SM
	size_t maximumThreadsPerMultiprocessor = deviceProps.maxThreadsPerMultiProcessor;

	/*we also determine the maximum resident blocks per SM. On
	 *architectures that are newer than 2.0 CC, we have 16 maximum
	 *architectures  resident blocks per SM, otherwise 8
	 */
	size_t maxBlocksPerMultiprocessor = (deviceProps.major > 2) ? 16 : 8;

	/*
	 * now we simply need to obtain the limit active blocks per SM with
	 * respect to those initial hardware considerations
	 */

	size_t threadCountLimit = 0;
	if (blockSize > deviceProps.maxThreadsPerBlock) {
		threadCountLimit = 0; // obviously if we have more threads per block than the device allows, we cannot run any blocks....
	} else {
		//if this is not the case
		threadCountLimit = maximumThreadsPerMultiprocessor / blockSize; // we simply see how many blocks of this size will fit on an SM
	}

	/*
	 * now our final limit is the minimum of those two limiters. Either we hit thread count first or block count
	 */
	size_t maxBlocksDeviceLimit = min2(maxBlocksPerMultiprocessor, threadCountLimit);

	return maxBlocksDeviceLimit;
}

#endif /* OCCUPANCYLIMITS_HPP_ */
