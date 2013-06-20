/**
 * OccupancyUtils.hpp
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYUTILS_HPP_
#define OCCUPANCYUTILS_HPP_

inline __host__
size_t getWarpAllocationGranularity(const cudaDeviceProp &properties)
{
  return (properties.major <= 1) ? 2 : 1;
}

inline __host__ size_t min2(size_t x, size_t y) {

	return (x < y) ? x : y;
}

inline __host__ size_t min3(size_t x, size_t y, size_t z) {
	return min2(z, min2(x, y));
}

inline __host__ int ceilTo(int x, int y) {
	int timesYRounded = (x + (y - 1)) / y;
	return y * timesYRounded;
}

inline __host__ int floorTo(int x, int y) {
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

// granularity of register allocation
inline __host__ size_t getRegisterAllocationGranularity(const cudaDeviceProp &devProps) {
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

size_t getSharedMemNeeded(const cudaFuncAttributes &kernelProps, const cudaDeviceProp &deviceProps) {

	// first we need to get the exact number of bytes statically allocated by the kernel (per block..)
	size_t sharedMemoryNeeded = kernelProps.sharedSizeBytes;
	size_t gran = getsharedMemoryGranularity(deviceProps); // we after that get the shared memory allocation granularity
	sharedMemoryNeeded = ceilTo(sharedMemoryNeeded, gran); // we now need to CEIL up to a multiple of that number
	return sharedMemoryNeeded;
}
#endif /* OCCUPANCYUTILS_HPP_ */
