/**
 * OccupancyData.hpp
 *
 *	This file contains two classes that are used in the process of tweaking kernel launch parameters
 *	in order to fit in particular ahrdware limitations.
 *
 *
 *  Created on: Jun 20, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef OCCUPANCYDATA_HPP_
#define OCCUPANCYDATA_HPP_
#define MAX_BLOCKS_PER_SM 8
#include "stdio.h"
#include <iostream>
#include <cmath>
#include "cuda_runtime.h"

/**
 * This class stores the amount of resources that are used by a particular block of
 * threads from a particular kernel. This includes total shared mem, threads and registers.
 */
class BlockUsage {

private:
	size_t sharedMem; // the shared memory used by the block
	size_t threads; // the number of threads that are used by the whole block
	size_t registers; // the numebr of registers used by the block
	size_t blocksPerSm; // the maximum active blocks that can be accomodated by a serial multiprocessor

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

	/*
	 * Just basic accessors and modifiers for the internal values
	 */
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
		output << "{block Usage} [sh_mem:" << lm.sharedMem << "] [thrs: " << lm.threads << "] [regs: " << lm.registers << "] [actv_blks: " << lm.blocksPerSm
				<< " ]";
		return output;
	}
};

/**
 * This class provides a data structure that stores the hardware limits that might be imposed on a kernel.
 * These limits serve as virtualization of the device real hardware capabilities and can be used to restrict
 * a kernel in order to use only particular fraction of the total resources present on the GPU
 *
 */
class KernelLimits {

private:
	size_t sharedMem; // the total shared memory available for the kernel
	size_t threads; // the total number of threads available for the kernel
	size_t registers; // the total number of registers available for the kernel
	size_t blocks; // the total number of thread blocks that the device can accommodate

	// a convinience function to round a number up to a size
	size_t roundToSize_t(double x) {
		return static_cast<size_t>(round(x));
	}

public:
	/**
	 * Default constructor that sets all limits to simply 1
	 */
	inline KernelLimits() {
		this->sharedMem = 1;
		this->threads = 1;
		this->registers = 1;
		this->blocks = 1;

	}

	/**
	 * This constructor allows device resource partitioning by specifying the fraction of
	 * resources that are to be used on the device. Using NVIDIA runtime API queries,
	 * limits are composed based on the actual hardware characteristics of the GPU that
	 * is installed on the machine
	 *
	 * @param shMemFrac fraction of the shred memory to be used
	 * @param threadsFrac fraction of the maximum threads to be used
	 * @param regsFrac fraction of the registers available to be used
	 * @param blocksFrac size of the max block size
	 * @param GPUConf .. the GPU configuration for the particular card
	 */
	inline KernelLimits(double shMemFrac, double threadsFrac, double regsFrac, double blocksFrac, const cudaDeviceProp& GPUConf) {
		// constructing limits based on data provided by the GPU hardware characteristics
		this->sharedMem = roundToSize_t((GPUConf.sharedMemPerBlock * GPUConf.multiProcessorCount) * shMemFrac);
		this->threads = roundToSize_t((GPUConf.maxThreadsPerMultiProcessor * GPUConf.multiProcessorCount) * threadsFrac);
		this->registers = roundToSize_t(((GPUConf.regsPerBlock * GPUConf.multiProcessorCount) * regsFrac));
		this->blocks = roundToSize_t(((GPUConf.maxThreadsPerBlock * MAX_BLOCKS_PER_SM * GPUConf.multiProcessorCount) * blocksFrac));

	}

	/**
	 * This constructor allows explicit declaration of values for the  different resource limits
	 * for maximum flexibility
	 *
	 * @param sharedMem total shared memory
	 * @param threads total num threads
	 * @param regs total num registers
	 * @param blocks total number of blocks
	 */
	inline KernelLimits(size_t sharedMem, size_t threads, size_t regs, size_t blocks) {
		this->sharedMem = sharedMem;
		this->threads = threads;
		this->registers = regs;
		this->blocks = blocks;
	}

	/*
	 * Accessors and modifyers
	 */
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

	/*
	 * A friend function to allow printing to ostreams
	 */
	inline friend std::ostream &operator<<(std::ostream &output, const KernelLimits &lm) {
		output << "Limits [SMem: " << lm.sharedMem << "] [Thrs: " << lm.threads << "] Regs " << lm.registers << " Blocks " << lm.blocks;
		return output;
	}

};

#endif /* OCCUPANCYDATA_HPP_ */
