/**
 * Macros.h
 *
 * The file contains functions that are useful when handling calls to the NVIDIA
 * driver API
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef MACROS_H_
#define MACROS_H_
#include <stdlib.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/**
 * Used to handle errors for kernel launches. Copied from NVIDIA examples
 *
 * @param code
 * @param file
 * @param line
 * @param abort
 */
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

/**
 * Returns the configuration properties of the particular GPU running on the machine
 *
 * @return the gpu configuration
 */
inline cudaDeviceProp getGPUConfiguration() {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	return props;
}

/**
 * Returns a double random number within a range
 *
 * @param min min threshold
 * @param max max threshold
 * @return
 */
inline double rnd(double min, double max) {

	double f = (double) rand() / RAND_MAX;
	return min + f * (max - min);
}

#endif /* MACROS_H_ */
