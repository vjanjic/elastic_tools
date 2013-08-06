/**
 * OccupancyUtils.hpp
 *
 * A set of functions performing mostly simple arithmetic operations,
 * that are useful when determining GPU limits
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYUTILS_HPP_
#define OCCUPANCYUTILS_HPP_
#pragma once
#include <cuda_runtime.h>

/**
 * Returns the warp allocation granularity depending on the architecture of the device
 *
 * @param properties cudaDeviceProp device properties
 * @return the warp allocation granularity
 */
inline size_t getWarpAllocationGranularity(const cudaDeviceProp &properties) {
	return (properties.major <= 1) ? 2 : 1;
}

/**
 * Returns the minimum of two numbers
 *
 * @param x number 1
 * @param y number 2
 *
 * @return the smaller number
 */
inline size_t min2(size_t x, size_t y) {

	return (x < y) ? x : y;
}

/**
 * Returns the minimum of three numbers
 *
 * @param x number 1
 * @param y number 2
 * @param z number 3
 *
 * @return the minimum
 */
inline size_t min3(size_t x, size_t y, size_t z) {
	return min2(z, min2(x, y));
}

/**
 * Performs a ceil operation where the value is ceiled up to multiple of some other value
 *
 * @param x the number to be ceiled
 * @param y the multiple
 *
 * @return ceiled value
 */

inline int ceilTo(int x, int y) {
	int timesYRounded = (x + (y - 1)) / y;
	return y * timesYRounded;
}

/**
 * Floors a value to a multiple of some other value
 *
 * @param x the value to be floored
 * @param y the multiple
 * @return the result
 */
inline int floorTo(int x, int y) {
	return y * (x / y);
}

/**
 * Depending on the particular GPU architecture, returns the shared memory allocation granularity
 * @param devProps the device configuration
 * @return the shared memory allocation granularity
 */
inline size_t getsharedMemoryGranularity(const cudaDeviceProp &devProps) {

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

/**
 * Returns the numebr of warp schedulers per SM
 *
 * @param devProps
 * @return
 */
// number of "sides" into which the multiprocessor is partitioned
inline size_t getNumWarpSchedulers(const cudaDeviceProp &devProps) {
	return (devProps.major < 3) ? devProps.major : 4;

}

/**
 * Returns the maximum number of blocks that can be accommodated on an SM
 *
 * @param devProps
 * @return
 */
inline size_t getMaxSMBlocks(const cudaDeviceProp &devProps) {
	return (devProps.major > 2) ? 16 : 8;
}

/**
 * Returns the register allocation granularity
 *
 * @param devProps
 * @return
 */
// granularity of register allocation
inline size_t getRegisterAllocationGranularity(const cudaDeviceProp &devProps) {
	switch (devProps.major) {
	case 1:
		return (devProps.minor <= 1) ? 256 : 512;
	case 2:
		return 64;

		/* no break */
	case 3:
		return 256;
	default:
		return 256; // unknown GPU; have to guess
	}
}

/**
 * Returns the shared memory needed by a particular kernel (per thread block)
 * @param kernelProps
 * @param deviceProps
 * @return
 */
inline size_t getSharedMemNeeded(const cudaFuncAttributes &kernelProps, const cudaDeviceProp &deviceProps) {

	// first we need to get the exact number of bytes statically allocated by the kernel (per block..)
	size_t sharedMemoryNeeded = kernelProps.sharedSizeBytes;
	size_t gran = getsharedMemoryGranularity(deviceProps); // we after that get the shared memory allocation granularity
	sharedMemoryNeeded = ceilTo(sharedMemoryNeeded, gran); // we now need to CEIL up to a multiple of that number
	return sharedMemoryNeeded;
}
#endif /* OCCUPANCYUTILS_HPP_ */
