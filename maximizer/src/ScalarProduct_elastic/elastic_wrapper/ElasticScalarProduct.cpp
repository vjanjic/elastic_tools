/**
 * ElasticScalarProduct.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ElasticScalarProduct.h"

ElasticScalarProduct::ElasticScalarProduct() :
		AbstractElasticKernel() {
	// TODO Auto-generated constructor stub

}

ElasticScalarProduct::ElasticScalarProduct(LaunchParameters& launchConfig, std::string name) :
		AbstractElasticKernel(launchConfig, name) {
}

void ElasticScalarProduct::initKernel() {
	h_A = (float *) malloc(DATA_SZ);
	h_B = (float *) malloc(DATA_SZ);
	h_C_CPU = (float *) malloc(RESULT_SZ);
	h_C_GPU = (float *) malloc(RESULT_SZ);

	CUDA_CHECK_RETURN((cudaMalloc((void ** )&d_A, DATA_SZ)));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_B, DATA_SZ));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_C, RESULT_SZ));

	srand(123);

	//Generating input data on CPU
	for (int i = 0; i < DATA_N; i++) {
		h_A[i] = RandFloat(0.0f, 1.0f);
		h_B[i] = RandFloat(0.0f, 1.0f);
	}

	CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice));

}

void ElasticScalarProduct::runKernel(cudaStream_t& streamToRunIn) {
	startSPKernel(gridConfig.getThreadsPerBlock(),gridConfig.getBlocksPerGrid(),d_C,d_A,d_B,VECTOR_N,ELEMENT_N,streamToRunIn);
}

cudaFuncAttributes ElasticScalarProduct::getKernelProperties() {
	return getSPKernelProperties();
}

void ElasticScalarProduct::freeResources() {

	CUDA_CHECK_RETURN(cudaFree(d_C));
	CUDA_CHECK_RETURN(cudaFree(d_B));
	CUDA_CHECK_RETURN(cudaFree(d_A));
	free(h_C_GPU);
	free(h_C_CPU);
	free(h_B);
	free(h_A);
}

float ElasticScalarProduct::RandFloat(float low, float high) {

	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + t * high;

}

size_t ElasticScalarProduct::getMemoryConsumption() {
	return (DATA_SZ * 2) + RESULT_SZ;
}

ElasticScalarProduct::~ElasticScalarProduct() {
	// TODO Auto-generated destructor stub
}

