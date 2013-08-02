/**
 * ElasticMatrixMultiplication.h
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ELASTICMATRIXMULTIPLICATION_H_
#define ELASTICMATRIXMULTIPLICATION_H_
#include "../../elastic_kernel/AbstractElasticKernel.hpp"
#include "../../misc/Macros.h"
#include "../GPU_code/Kernel_Starter_MM.h"

class ElasticMatrixMultiplication: public AbstractElasticKernel {
private:
	int matrixWidth;
	float* d_M;
	float* d_N;
	float* d_P;


public:
	ElasticMatrixMultiplication();
	ElasticMatrixMultiplication(LaunchParameters &launchConfig, std::string name, int matrixSize);

	void initKernel();
	void runKernel(cudaStream_t &streamToRunIn);
	cudaFuncAttributes getKernelProperties();
	void freeResources();
	size_t getMemoryConsumption();
	virtual ~ElasticMatrixMultiplication();
};

#endif /* ELASTICMATRIXMULTIPLICATION_H_ */
