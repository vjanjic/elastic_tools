/**
 * OccupancyData.hpp
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYDATA_HPP_
#define OCCUPANCYDATA_HPP_
#include "stdio.h"
#include <iostream>
#include <cmath>
#include "cuda_runtime.h"

/*typedef struct {
 size_t sharedMemory;
 size_t numThreads;
 size_t numRegisters;
 size_t blocksPerSM;
 } BlockUsage;*/

class BlockUsage {

private:
	size_t sharedMem;
	size_t threads;
	size_t registers;
	size_t blocksPerSm;

public:
	inline BlockUsage() {
		this->sharedMem = 1;
		this->threads = 1;
		this->registers = 1;
		this->blocksPerSm = 1;
	}

	inline BlockUsage(size_t sharedMem, size_t thrs, size_t registers, size_t blocks) {
		this->sharedMem = sharedMem;
		this->threads = thrs;
		this->registers = registers;
		this->blocksPerSm = blocks;

	}

	inline size_t getSharedMem() {
		return this->sharedMem;
	}
	inline size_t getNumThreads() {
		return this->threads;
	}
	inline size_t getNumRegisters() {
		return this->registers;
	}
	inline size_t getNumBlocksPerSM() {
		return this->blocksPerSm;
	}

	inline void setSharedMem(size_t shMem) {
		this->sharedMem = shMem;
	}
	inline void setNumThreads(size_t numThrs) {
		this->threads = numThrs;
	}
	inline void setNumRegisters(size_t numRegs) {
		this->registers = numRegs;
	}
	inline void setNumBlocks(size_t blocks) {
		this->blocksPerSm = blocks;
	}
	inline friend std::ostream &operator<<(std::ostream &output, const BlockUsage &lm) {
		output << "{block Usage} [sh_mem:" << lm.sharedMem << "] [thrs: " << lm.threads << "] [regs: " << lm.registers << "] [actv_blks: " << lm.blocksPerSm  << " ]";
		return output;
	}
};

class KernelLimits {

private:
	size_t sharedMem;
	size_t threads;
	size_t registers;
	size_t blocks;

	size_t roundToSize_t(double x) {
		return static_cast<size_t>(round(x));
	}

public:

	inline KernelLimits() {
		this->sharedMem = 1;
		this->threads = 1;
		this->registers = 1;
		this->blocks = 1;

	}

	inline KernelLimits(double shMemFrac, double threadsFrac, double regsFrac, double blocksFrac, const cudaDeviceProp& GPUConf) {

		this->sharedMem = roundToSize_t((GPUConf.sharedMemPerBlock * GPUConf.multiProcessorCount) * shMemFrac);
		this->threads =  roundToSize_t((GPUConf.maxThreadsPerMultiProcessor * GPUConf.multiProcessorCount) * threadsFrac);
		this->registers = roundToSize_t(((GPUConf.regsPerBlock * GPUConf.multiProcessorCount) * regsFrac));
		this->blocks = roundToSize_t(((8 * GPUConf.multiProcessorCount) * blocksFrac));
	}

	inline KernelLimits(size_t sharedMem, size_t threads, size_t regs, size_t blocks) {
		this->sharedMem = sharedMem;
		this->threads = threads;
		this->registers = regs;
		this->blocks = blocks;
	}

	inline size_t getSharedMem() {
		return this->sharedMem;
	}
	inline size_t getNumThreads() {
		return this->threads;
	}
	inline size_t getNumRegisters() {
		return this->registers;
	}
	inline size_t getNumBlocks() {
		return this->blocks;
	}

	inline void setSharedMem(size_t shMem) {
		this->sharedMem = shMem;
	}
	inline void setNumThreads(size_t numThrs) {
		this->threads = numThrs;
	}
	inline void setNumRegisters(size_t numRegs) {
		this->registers = numRegs;
	}
	inline void setNumBlocks(size_t blocks) {
		this->blocks = blocks;
	}
	inline friend std::ostream &operator<<(std::ostream &output, const KernelLimits &lm) {
		output << "Limits [SMem: " << lm.sharedMem << "] [Thrs: " << lm.threads << "] Regs " << lm.registers << " Blocks " << lm.blocks;
		return output;
	}

};

#endif /* OCCUPANCYDATA_HPP_ */
