/**
 * ElasticChunker.h
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ELASTICCHUNKER_H_
#define ELASTICCHUNKER_H_

#include "../../elastic_kernel/AbstractElasticKernel.hpp"
#include "../GPU_code/KernelStarter_CS.h"
#include "../GPU_code/rabin_fingerprint/RabinFingerprint.h"
#include "../GPU_code/ResourceManagement.h"
#include <stdio.h>
#include <cuda_runtime.h>


class ElasticChunker: public AbstractElasticKernel {
private:
	BYTE* dataBuffer_d;
	rabinData* rabinData_d;
	bitFieldArray results_d;
	size_t dataSize;
public:
	ElasticChunker();
	ElasticChunker(LaunchParameters &launchConfig, std::string name);
	virtual ~ElasticChunker();
	void initKernel();
	void runKernel(cudaStream_t &streamToRunIn);
	cudaFuncAttributes getKernelProperties();
	size_t getMemoryConsumption();
	void freeResources();

};

#endif /* ELASTICCHUNKER_H_ */
