/**
 * ElasticVectorAddition.h
 *
 *  Created on: Jul 29, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ELASTICVECTORADDITION_H_
#define ELASTICVECTORADDITION_H_

#include "../../elastic_kernel/AbstractElasticKernel.hpp"
#include "../../misc/Macros.h"
#include "../GPU_code/Kernel_Starter_VA.h"

class ElasticVectorAddition: public AbstractElasticKernel {
private:
	int* a;
	int* b;
	int numElems;

public:
	ElasticVectorAddition();
	ElasticVectorAddition(LaunchParameters &launchConfig, std::string name);

	void initKernel();
	void runKernel(cudaStream_t &streamToRunIn);
	cudaFuncAttributes getKernelProperties();
	void freeResources();

	virtual ~ElasticVectorAddition();
};

#endif /* ELASTICVECTORADDITION_H_ */
