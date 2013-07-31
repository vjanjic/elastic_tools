/**
 * ElasticChunker.cpp
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ElasticChunker.h"

ElasticChunker::ElasticChunker() :
		AbstractElasticKernel(), dataSize(67108864), rabinData_d(0), dataBuffer_d(0), results_d(0) {

}

ElasticChunker::ElasticChunker(LaunchParameters &launchConfig, std::string name) :
		AbstractElasticKernel(launchConfig,name), dataSize(67108864), rabinData_d(0), dataBuffer_d(0), results_d(0) {

}

ElasticChunker::~ElasticChunker() {

}

void ElasticChunker::initKernel() {

	BYTE* hostBuffer = (BYTE*) malloc(sizeof(BYTE) * dataSize);
	srand(2);
	for (int var = 0; var < dataSize; ++var) {
		hostBuffer[var] = (BYTE) rand() % 256;
	}

	CUDA_CHECK_RETURN(cudaMalloc(&dataBuffer_d, sizeof(BYTE) * dataSize));
	CUDA_CHECK_RETURN(cudaMemcpy(dataBuffer_d, hostBuffer, dataSize * sizeof(BYTE), cudaMemcpyHostToDevice));

	rabinData hostData;

	initWindow(&hostData, 0xbfe6b8a5bf378d83);
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &rabinData_d, sizeof(rabinData)));
	CUDA_CHECK_RETURN(cudaMemcpy(rabinData_d, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));

	int numberOfBitWordsNeeded = getSizeOfBitArray(dataSize);
	this->results_d = createBitFieldArrayOnDevice(numberOfBitWordsNeeded);
	free(hostBuffer);
}



cudaFuncAttributes ElasticChunker::getKernelProperties() {
	return getChunkingKernelProperties();
}

void ElasticChunker::runKernel(cudaStream_t& streamToRunIn) {
	size_t totalNumThreads = this->gridConfig.getNumTotalThreads();

	size_t workPerThread = this->dataSize / totalNumThreads;



	startCreateBreakpointsKernel(gridConfig.getThreadsPerBlock(), gridConfig.getBlocksPerGrid(), this->rabinData_d, this->dataBuffer_d, dataSize,
			this->results_d, totalNumThreads, workPerThread, 512,streamToRunIn);
}

size_t ElasticChunker::getMemoryConsumption() {
	return (sizeof(BYTE) * dataSize) + sizeof(rabinData) + (getSizeOfBitArray(dataSize) * 32);
}

void ElasticChunker::freeResources() {
	freeCudaResource(this->dataBuffer_d);
	freeCudaResource(this->rabinData_d);
	freeCudaResource(this->results_d);
}
