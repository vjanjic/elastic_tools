/**
 * ElasticVectorAddition.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ElasticVectorAddition.h"
#define VEC_LEN 10000000;


ElasticVectorAddition::ElasticVectorAddition() :
		AbstractElasticKernel() {
	this->numElems = VEC_LEN;
	// TODO Auto-generated constructor stub

}

ElasticVectorAddition::ElasticVectorAddition(LaunchParameters& launchConfig, std::string name) :
		AbstractElasticKernel(launchConfig, name) {
	this->numElems = VEC_LEN;

}

void ElasticVectorAddition::initKernel() {
	int* A_host = (int*) malloc(sizeof(int) * this->numElems);
	int* B_host = (int*) malloc(sizeof(int) * this->numElems);

	srand(2);
	for (int var = 0; var < this->numElems; ++var) {
		A_host[var] = rand();
		B_host[var] = rand();

	}

	CUDA_CHECK_RETURN(cudaMalloc(&a, sizeof(int) * this->numElems));
	CUDA_CHECK_RETURN(cudaMalloc(&b, sizeof(int) * this->numElems));

	CUDA_CHECK_RETURN(cudaMemcpy(a, A_host, sizeof(int) * this->numElems, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(b, B_host, sizeof(int) * this->numElems, cudaMemcpyHostToDevice));

	free(A_host);
	free(B_host);

}

void ElasticVectorAddition::runKernel(cudaStream_t& streamToRunIn) {
	startAddVectorsKernel(gridConfig.getThreadsPerBlock(), gridConfig.getBlocksPerGrid(), this->a, this->b, this->numElems, streamToRunIn);

}

cudaFuncAttributes ElasticVectorAddition::getKernelProperties() {

	return getVectorAddKernelProperties();
}

void ElasticVectorAddition::freeResources() {
	CUDA_CHECK_RETURN(cudaFree(this->a));
	CUDA_CHECK_RETURN(cudaFree(this->b));

}

ElasticVectorAddition::~ElasticVectorAddition() {
	// TODO Auto-generated destructor stub
}

