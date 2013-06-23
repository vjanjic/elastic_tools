/**
 * ResourceLimits.hpp
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef RESOURCELIMITS_HPP_
#define RESOURCELIMITS_HPP_

typedef struct {
	size_t numBlocks;
	size_t shMem;
	size_t threads;
	size_t registers;
} Resourcelimits;

typedef struct {
	size_t sharedMem;
	size_t numThreads;
	size_t numRegisters;
} ThreadBlockUsage;

typedef struct {
	size_t sharedMem;
	size_t numRegisters;
	size_t threadsPerBlock;
	size_t numBlocks;
} KernelInfo;

typedef struct {
	size_t physThPerBlock;
	size_t physBlocks;
} PhysicalGrid;

PhysicalGrid getConfiguration() {
	PhysicalGrid grid;

	return grid;
}

typedef struct {
	size_t sharedMemory;
	size_t numThreads;
	size_t numRegisters;
	size_t blocksPerSM;
} BlockUsage;



#endif /* RESOURCELIMITS_HPP_ */
