/**
 * Kernel_Starter.h
 *
 *  Created on: Jul 29, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNEL_STARTER_SP_H_
#define KERNEL_STARTER_SP_H_
#include "stdio.h"
#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>

extern "C" void startSPKernel(size_t threads, size_t blocks, float *d_C, float *d_A, float *d_B, int vectorN, int elementN, cudaStream_t& stream);

extern "C" cudaFuncAttributes getSPKernelProperties();

#endif /* KERNEL_STARTER_SP_H_ */
