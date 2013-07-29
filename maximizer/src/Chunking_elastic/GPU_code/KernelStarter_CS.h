/**
 * KernelStarter.h
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELSTARTER_CH_H_
#define KERNELSTARTER_CH_H_
#include "DedupDefines.h"
#include "rabin_fingerprint/RabinData.h"
#include "BitFieldArray.h"
extern "C" void startCreateBreakpointsKernel(int blocksSize, int numBlocks, rabinData* deviceRabin, BYTE* deviceData, int dataLen, bitFieldArray results,
		int threadsUsed, int workPerThread, int D, cudaStream_t stream);

extern "C"  cudaFuncAttributes getChunkingKernelProperties();

#endif /* KERNELSTARTER_CH_H_ */
