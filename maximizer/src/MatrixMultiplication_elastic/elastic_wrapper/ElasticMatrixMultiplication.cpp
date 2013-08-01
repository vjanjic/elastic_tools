/**
 * ElasticMatrixMultiplication.cpp
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ElasticMatrixMultiplication.h"

ElasticMatrixMultiplication::ElasticMatrixMultiplication() :
		AbstractElasticKernel() {
	this->matrixWidth = 4096;
	this->memConsumption = matrixWidth * matrixWidth * sizeof(float) * 3;
}

ElasticMatrixMultiplication::ElasticMatrixMultiplication(LaunchParameters& launchConfig, std::string name) :
		AbstractElasticKernel(launchConfig, name) {
	this->matrixWidth = 4096;
	this->memConsumption = matrixWidth * matrixWidth * sizeof(float) * 3;

}

void ElasticMatrixMultiplication::initKernel() {

	int matrix_size = matrixWidth * matrixWidth * sizeof(float);

	float* M = (float*) malloc(sizeof(float) * matrix_size);
	float* N = (float*) malloc(sizeof(float) * matrix_size);
	CUDA_CHECK_RETURN(cudaMalloc(&d_M, matrix_size));
	CUDA_CHECK_RETURN(cudaMemcpy(d_M, M, matrix_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc(&d_N, matrix_size));
	CUDA_CHECK_RETURN(cudaMemcpy(d_N, N, matrix_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc(&d_P, matrix_size));

	free(M);
	free(N);

}

void ElasticMatrixMultiplication::runKernel(cudaStream_t& streamToRunIn) {
	int tileW = matrixWidth / gridConfig.getNumTotalThreads();
	startMMKernel(gridConfig.getThreadsPerBlock(), gridConfig.getBlocksPerGrid(), d_M, d_N, d_P, matrixWidth, tileW, streamToRunIn);
}

cudaFuncAttributes ElasticMatrixMultiplication::getKernelProperties() {
	return getMMKernelProperties();
}

void ElasticMatrixMultiplication::freeResources() {
	CUDA_CHECK_RETURN(cudaFree(d_M));
	CUDA_CHECK_RETURN(cudaFree(d_N));
	CUDA_CHECK_RETURN(cudaFree(d_P));
}

size_t ElasticMatrixMultiplication::getMemoryConsumption() {
	return this->memConsumption;

}

ElasticMatrixMultiplication::~ElasticMatrixMultiplication() {
	// TODO Auto-generated destructor stub
}

