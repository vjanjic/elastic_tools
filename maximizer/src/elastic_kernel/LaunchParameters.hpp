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
#include <iostream>

class LaunchParameters {

private:
	size_t blockX;
	size_t blockY;
	size_t blockZ;

	size_t gridX;
	size_t gridY;

public:
	inline LaunchParameters() {
		this->blockX = 1;
		this->blockY = 1;
		this->blockZ = 1;
		this->gridX = 1;
		this->gridY = 1;
	}

	inline LaunchParameters(size_t thrsPerBlock, size_t blksPerGrid) {
		this->blockX = thrsPerBlock;
		this->blockY = 1;
		this->blockZ = 1;
		this->gridX = blksPerGrid;
		this->gridY = 1;
	}

	inline LaunchParameters(size_t blockX, size_t blockY, size_t blockZ, size_t gridX, size_t gridY) {
		this->blockX = blockX;
		this->blockY = blockY;
		this->blockZ = blockZ;
		this->gridX = gridX;
		this->gridY = gridY;
	}

	inline ~LaunchParameters() {

	}

	inline void setDimensions(size_t threads, size_t blocks) {
		this->blockX = threads;
		this->blockY = 1;
		this->blockZ = 1;
		this->gridX = blocks;
		this->gridY = 1;
	}

	inline size_t getNumTotalThreads() {
		return this->blockX * this->blockY * this->blockZ * this->gridX * this->gridY;
	}

	inline size_t getThreadsPerBlock() {
		return this->blockX * this->blockY * this->blockZ;
	}

	inline size_t getBlocksPerGrid() {
		return this->gridX * this->gridY;
	}

	inline friend std::ostream &operator<<(std::ostream &output, const LaunchParameters &pp) {
		output << "[" << pp.blockX << " x " << pp.blockY << " x " << pp.blockZ << "] [" << pp.gridX << " x " << pp.gridY << "]";
		return output;
	}

};

#endif /* LOGICALCONFIGURATION_HPP_ */
