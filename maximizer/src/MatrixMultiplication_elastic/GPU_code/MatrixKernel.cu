/**
 * MatrixKernel.cu
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "stdio.h"
#include "../../misc/Macros.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline __device__ int getID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width, int tileWidth, int totalThrs) {

	int tID = getID();

	if (tID != totalThrs - 1) {
		int start_row = (tID * tileWidth);
		int end_row = (start_row + tileWidth);
		int start_col = (tID * tileWidth);
		int end_col = (start_col + tileWidth);

		for (int row = start_row; row < end_row; row++) {
			for (int col = start_col; col < end_col; col++) {
				float P_val = 0;
				for (int k = 0; k < Width; ++k) {
					float M_elem = d_M[row * Width + k];
					float N_elem = d_N[k * Width + col];
					P_val += M_elem * N_elem;
				}
				d_P[row * Width + col] = P_val;
			}
		}
	}

}

extern "C" void startMMKernel(size_t threads, size_t blocks, float* d_M, float* d_N, float* d_P, int mtrxWidth, int tileWidth, int totalThrs,
		cudaStream_t stream) {
	MatrixMulKernel<<<blocks, threads, 0, stream>>>(d_M, d_N, d_P, mtrxWidth, tileWidth, totalThrs);
	gpuErrchk(cudaGetLastError());

}

extern "C" cudaFuncAttributes getMMKernelProperties() {
	cudaFuncAttributes attributes;
	cudaFuncGetAttributes(&attributes, MatrixMulKernel);
	return attributes;
}
