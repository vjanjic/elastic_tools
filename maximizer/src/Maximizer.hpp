/**
 * Maximizer.hpp
 *
 *  Created on: Jun 19, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef MAXIMIZER_HPP_
#define MAXIMIZER_HPP_
inline __host__ int roundUpTo_Y(int x, int y) {
	int timesYRounded = (x + (y - 1)) / y;
	return y * timesYRounded;
}

inline __host__ int roundDownTo_Y(int x, int y) {
	return y * (x / y);
}

inline __host__ size_t getsharedMemoryGranularity(const cudaDeviceProp &devProps) {

	/*
	 * according to the architecture shared memory is allocted
	 * in batches of certain size. We need to know this size
	 * in order to round up to it.
	 */
	switch (devProps.major) {
	case 1:
		return 512;
	case 2:
		return 128;
	case 3:
		return 256;
	default:
		return 128;
	}
}

// number of "sides" into which the multiprocessor is partitioned
inline __host__ size_t getNumWarpSchedulers(const cudaDeviceProp &devProps) {
	return (devProps.major < 3) ? devProps.major : 4;

}

inline __host__ size_t getMaxSMBlocks(const cudaDeviceProp &devProps) {
	return (devProps.major > 2) ? 16 : 8;
}

inline __host__ size_t min(size_t x, size_t y) {

	return (x < y) ? x : y;
}

// granularity of register allocation
inline __host__ size_t getRegisterAllocationGranularity(const cudaDeviceProp &devProps, const size_t regsPerThread) {
	switch (devProps.major) {
	case 1:
		return (devProps.minor <= 1) ? 256 : 512;
	case 2:
		switch (regsPerThread) {
		case 21:
		case 22:
		case 29:
		case 30:
		case 37:
		case 38:
		case 45:
		case 46:
			return 128;
		default:
			return 64;
		}
		/* no break */
	case 3:
		return 256;
	default:
		return 256; // unknown GPU; have to guess
	}
}

inline __host__ size_t getMaxActiveBlocksPerSM(const cudaDeviceProp &properties, const cudaFuncAttributes &attributes, size_t blockSize) {

#ifdef DEBUG
	printf("This is a debug version...");
#endif

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	 * First we start by examining the actual hardware limits
	 * of the GPU with respect to maximum active blocks of
	 * size N per SM....
	 */

	// we first need the maximum number of resident threads per SM
	size_t maximumThreadsPerMultiprocessor = properties.maxThreadsPerMultiProcessor;

	/*we also determine the maximum resident blocks per SM. On
	 *architectures that are newer than 2.0 CC, we have 16 maximum
	 *architectures  resident blocks per SM, otherwise 8
	 */
	size_t maxBlocksPerMultiprocessor = (properties.major > 2) ? 16 : 8;

	/*
	 * now we simply need to obtain the limit active blocks per SM with
	 * respect to those initial hardware considerations
	 */

	size_t threadCountLimit = 0;
	if (blockSize > properties.maxThreadsPerBlock) {
		threadCountLimit = 0; // obviously if we have more threads per block than the device allows, we cannot run any blocks....
	} else {
		//if this is not the case
		threadCountLimit = maximumThreadsPerMultiprocessor / blockSize; // we simply see how many blocks of this size will fit on an SM
	}

	/*
	 * now our final limit is the minimum of those two limiters. Either we hit thread count first or block count
	 */
	size_t maxBlocksDeviceLimit = min(maxBlocksPerMultiprocessor, threadCountLimit);
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * Now we need to consider how many blocks can we really have active due to the limit of our shared memory that is requested.
	 */

	// first we need to get the exact number of bytes statically allocated by the kernel (per block..)
	size_t sharedMemoryNeeded = attributes.sharedSizeBytes;
	size_t gran = getsharedMemoryGranularity(properties); // we after that get the shared memory allocation granularity
	sharedMemoryNeeded = roundUpTo_Y(sharedMemoryNeeded, gran); // we now need to CEIL up to a multiple of that number

	size_t maxBlocksSMLimit; // now we are ready to check out limit posed by shared memory
	if (sharedMemoryNeeded > 0) {
		maxBlocksSMLimit = properties.sharedMemPerBlock / sharedMemoryNeeded;
	} else {
		// else we know we do not have any limit on that, so we set the max to be the maximum num of active blocks per SM
		maxBlocksSMLimit = maxBlocksPerMultiprocessor;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * Now we need to calculate the limits due to register pressure
	 */

	// we first get the register allocation granularity and the warp allocation one
	size_t registerAlocationGranularity = getRegisterAllocationGranularity(properties, attributes.numRegs);
	size_t warpAllocationGranularity = (properties.major > 1) ? 1 : 2;

	/*
	 * We here calculate the number of warps needed for this number of threads per CTA.
	 * This is done by rounded devision to the number of threads per warp (which is the warp size)
	 * After that we check the compute capability.
	 */
	size_t numerWarpsNeeded = (blockSize + (properties.warpSize - 1)) / properties.warpSize; // we need to devide and round UP to warpsize
	/*
	 * here we CEIL up to 1 or 2, depending on the compute capability of the GPU
	 */
	numerWarpsNeeded = roundUpTo_Y(numerWarpsNeeded, warpAllocationGranularity); // again CEIL up to the war allocation granularity, depending on architecture;

	size_t maxBlocksRegisterLimit;
	if (properties.major <= 1) {
		/*
		 * We know that devices of compute capability of 1.x allocate registers per block.
		 * THerefore, number of registers per block would be the number of warps times
		 * times the num of registers per thread, all of that CEILED up to the register
		 * allocation size.
		 */
		size_t registersNeeded = attributes.numRegs * properties.warpSize * numerWarpsNeeded;
		registersNeeded = roundUpTo_Y(registersNeeded, registerAlocationGranularity);
		if (registersNeeded > 0) {
			/*if the needed registers is more than 0  we get the limit of maximum blocks
			 *with respect to the register pressure, simply by dividing the avaible
			 *registers per block by the number of registers we need to see how many
			 *we can fit in.
			 *
			 */
			maxBlocksRegisterLimit = properties.regsPerBlock / registersNeeded;
		} else {
			// if we do not need any registers, simply set the max to the max of blocks per SM
			maxBlocksRegisterLimit = maxBlocksPerMultiprocessor;
		}
	} else {
		/*
		 * In case our device is of higher compute capability than one, we need to consider that
		 * registers are allocated per warps. So the number of registers per warp would be the number of
		 * registers per thread * number of threads per warp * number of warps, CEILED_UP to the register
		 * allocation unit
		 */
		size_t registersPerWarp = roundUpTo_Y(attributes.numRegs * properties.warpSize, registerAlocationGranularity);
		size_t numberOfWarpSchedulersPerSM = getNumWarpSchedulers(properties); // depending on architecture (Kepler has a quad warp scheduler)
		size_t registersPerScheduler = properties.regsPerBlock / numberOfWarpSchedulersPerSM;
		if (registersPerWarp > 0) {
			maxBlocksRegisterLimit = ((registersPerScheduler / registersPerWarp) * numberOfWarpSchedulersPerSM) / numerWarpsNeeded;
		} else {
			maxBlocksRegisterLimit = maxBlocksPerMultiprocessor;
		}
	}
	return min(maxBlocksSMLimit, min(maxBlocksDeviceLimit, maxBlocksRegisterLimit));
}

inline __host__ size_t optimalBlockSize(const cudaDeviceProp &properties, const cudaFuncAttributes &attributes) {

	size_t maxBlockSize = min(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);

	size_t maximumBlockSizeAchieved = 0;
	size_t highestOccupancyAchieved = 0;

	for (size_t currentBlockSize = maxBlockSize; currentBlockSize != 0; currentBlockSize -= properties.warpSize) {

		size_t numBlocksForCTA = getMaxActiveBlocksPerSM(properties, attributes, currentBlockSize);
		size_t totalNumThreadsForBlockSize = currentBlockSize * numBlocksForCTA;
		printf("Blocksize:maximiser %d\n", totalNumThreadsForBlockSize);

		if (totalNumThreadsForBlockSize > highestOccupancyAchieved) {
			maximumBlockSizeAchieved = currentBlockSize;
			highestOccupancyAchieved = totalNumThreadsForBlockSize;
		}

		if (highestOccupancyAchieved == properties.maxThreadsPerMultiProcessor)
			break;


	}

	return maximumBlockSizeAchieved;
}

#endif /* MAXIMIZER_HPP_ */
