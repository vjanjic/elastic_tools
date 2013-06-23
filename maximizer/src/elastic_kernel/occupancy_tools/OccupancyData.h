/**
 * OccupancyData.hpp
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYDATA_HPP_
#define OCCUPANCYDATA_HPP_
#include "stdio.h"

typedef struct {
	size_t sharedMemory;
	size_t numThreads;
	size_t numRegisters;
	size_t blocksPerSM;
} BlockUsage;

inline void printUsage(BlockUsage us) {
	printf("----------------------\n");

	printf("Shared mem %d\n", us.sharedMemory);
	printf("Num thrs %d\n", us.numThreads);
	printf("Num regs %d\n", us.numRegisters);
	printf("Resident blks %d\n", us.blocksPerSM);
	printf("----------------------\n");

}

typedef struct {
	size_t sharedMem;
	size_t threads;
	size_t registers;
	size_t blocks;
} KernelLimits;

#endif /* OCCUPANCYDATA_HPP_ */
