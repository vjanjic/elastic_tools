/**
 * Kernel_Starter_VA.h
 *
 *  Created on: Jul 29, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNEL_STARTER_VA_H_
#define KERNEL_STARTER_VA_H_
#include "string.h"

extern "C" void startAddVectorsKernel(size_t threads, size_t blocks, int *A, int *B, int numElements, cudaStream_t stream);


extern "C"  cudaFuncAttributes getVectorAddKernelProperties();

#endif /* KERNEL_STARTER_VA_H_ */
