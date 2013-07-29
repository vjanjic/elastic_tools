/**
 * Kernel_Starter.h
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNEL_STARTER_BS_H_
#define KERNEL_STARTER_BS_H_

extern "C" void startBSKernel(size_t threads, size_t blocks,float* callResults, float* putResults, float* stockPrice, float* strike, float* years, float riskFree, float vol, int optN, cudaStream_t stream);

extern "C"  cudaFuncAttributes getBSKernelProperties();


#endif /* KERNEL_STARTER_BS_H_ */
