/**
 * KernelStarter_MM.h
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELSTARTER_MM_H_
#define KERNELSTARTER_MM_H_


extern "C" void startMMKernel(size_t threads, size_t blocks, float* d_M, float* d_N, float* d_P, int mtrxWidth, int tileWidth,int  totalThrs, cudaStream_t stream);


extern "C"  cudaFuncAttributes getMMKernelProperties();


#endif /* KERNELSTARTER_MM_H_ */
