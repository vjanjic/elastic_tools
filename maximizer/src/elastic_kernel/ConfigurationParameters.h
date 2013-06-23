/**
 * LogicalConfiguration.hpp
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef LOGICALCONFIGURATION_HPP_
#define LOGICALCONFIGURATION_HPP_
#include "stddef.h"
#include "stdio.h"

typedef struct {
	size_t grdDim_X;
	size_t grdDim_Y;
} gridParams_logical;

typedef struct {
	size_t blkDim_X;
	size_t blkDim_Y;
	size_t blkDim_Z;

} blockParams_logical;

typedef struct {
	size_t threadsPerBlock;
	size_t blocksPerGrid;

} PhysicalConfiguration;


inline void printHysicalConfig(PhysicalConfiguration ph) {

	printf("Blocks %d\n", ph.blocksPerGrid);
		printf("Threads %d\n", ph.threadsPerBlock);

}

inline gridParams_logical getLogicalGrid(size_t dx, size_t dy) {
	gridParams_logical result;
	result.grdDim_X = dx;
	result.grdDim_Y = dy;
	return result;
}

inline blockParams_logical getLogicalBlock(size_t dx, size_t dy, size_t dz) {
	blockParams_logical result;
	result.blkDim_X = dx;
	result.blkDim_Y = dy;
	result.blkDim_Z = dz;
	return result;
}

inline PhysicalConfiguration getPhysicalConfiguration(size_t blocks, size_t threads) {
	PhysicalConfiguration result;
	result.blocksPerGrid = blocks;
	result.threadsPerBlock = threads;
	return result;
}
#endif /* LOGICALCONFIGURATION_HPP_ */
