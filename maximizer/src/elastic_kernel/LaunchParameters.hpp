/**
 * LaunchParameters.hpp
 *
 * This class provides a data structure to hold the launch parameters
 * needed by AbstractElasticKernel instances. Along with that several
 * modifiers and accessors are provided.
 *
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef LOGICALCONFIGURATION_HPP_
#define LOGICALCONFIGURATION_HPP_
#include "stddef.h"
#include "stdio.h"
#include <iostream>

class LaunchParameters {

private:
	// all the needed vars for launching a kernel
	size_t blockX;
	size_t blockY;
	size_t blockZ;

	size_t gridX;
	size_t gridY;

public:
	/**
	 * Default constructor just defaults everything to 1
	 */
	inline LaunchParameters() {
		this->blockX = 1;
		this->blockY = 1;
		this->blockZ = 1;
		this->gridX = 1;
		this->gridY = 1;
	}

	/**
	 * Constructor for a little more control. This is for a one dimensional configuration
	 *
	 * @param thrsPerBlock the number of threads per block
	 * @param blksPerGrid the total number of blocks in the grid
	 */
	inline LaunchParameters(size_t thrsPerBlock, size_t blksPerGrid) {
		this->blockX = thrsPerBlock;
		this->blockY = 1;
		this->blockZ = 1;
		this->gridX = blksPerGrid;
		this->gridY = 1;
	}

	/**
	 * This constructor provides the functionality to specify a three dimensional
	 * configuration for the block and two dimensional for the grid.
	 *
	 * @param blockX block dim in X
	 * @param blockY block dim in Y
	 * @param blockZ block dim in Z
	 * @param gridX  grid dim in X
	 * @param gridY  grid dim in Y
	 */
	inline LaunchParameters(size_t blockX, size_t blockY, size_t blockZ, size_t gridX, size_t gridY) {
		this->blockX = blockX;
		this->blockY = blockY;
		this->blockZ = blockZ;
		this->gridX = gridX;
		this->gridY = gridY;
	}

	inline ~LaunchParameters() {

	}

	/**
	 * Setter for the one dimensional sizes
	 *
	 * @param threads number of threads per block
	 * @param blocks number of blocks per grid
	 */
	inline void setDimensions(size_t threads, size_t blocks) {
		this->blockX = threads;
		this->blockY = 1;
		this->blockZ = 1;
		this->gridX = blocks;
		this->gridY = 1;
	}

	/**
	 * Setter for just the block number per grid
	 *
	 * @param blocks num blocks per grid
	 */
	inline void setBlocks(size_t blocks) {
		this->gridX = blocks;
		this->gridY = 1;
	}

	/**
	 * Setter for the number of threads per block
	 *
	 * @param threads num threads per block
	 */
	inline void setThreads(size_t threads) {
		this->blockX = threads;
		this->blockY = 1;
		this->blockZ = 1;
	}

	/**
	 * Retrieves the total number of threads for the whole launch config
	 * @return
	 */
	inline size_t getNumTotalThreads() {
		return this->blockX * this->blockY * this->blockZ * this->gridX * this->gridY;
	}
	/**
	 * Retrieves the number of threads per block for this launch config
	 * @return
	 */
	inline size_t getThreadsPerBlock() {
		return this->blockX * this->blockY * this->blockZ;
	}

	/**
	 * Retrieves the totla number of blocks for the grid
	 * @return
	 */
	inline size_t getBlocksPerGrid() {
		return this->gridX * this->gridY;
	}
	// just used to print to an ostream..
	inline friend std::ostream &operator<<(std::ostream &output, const LaunchParameters &pp) {
		output << "[" << pp.blockX << " x " << pp.blockY << " x " << pp.blockZ << "] [" << pp.gridX << " x " << pp.gridY << "]";
		return output;
	}

};

#endif /* LOGICALCONFIGURATION_HPP_ */
