/**
 * OccupancyData.hpp
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYDATA_HPP_
#define OCCUPANCYDATA_HPP_
typedef struct {
	size_t sharedMemory;
	size_t numThreads;
	size_t numRegisters;
	size_t blocksPerSM;
} BlockUsage;

#endif /* OCCUPANCYDATA_HPP_ */
