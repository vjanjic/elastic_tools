/**
 * ElasticBSPricer.h
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ELASTICBSPRICER_H_
#define ELASTICBSPRICER_H_
#include "../../../abstract_elastic_kernel/AbstractElasticKernel.hpp"
#include "../../../misc/Macros.h"
#include "../GPU_code/Kernel_Starter_BS.h"

#include <stdio.h>
#include <stdlib.h>

class ElasticBSPricer: public AbstractElasticKernel {

private:
	//'d_' prefix - GPU (device) memory space
	float *d_CallResult;
	float *d_PutResult;
	float *d_StockPrice;
	float *d_OptionStrike;
	float *d_OptionYears;

	const static int OPT_N = 40000000;
	const static int NUM_ITERATIONS = 2048;
	const  static int OPT_SZ = OPT_N * sizeof(float);
	const static float RISKFREE = 0.02f;
	const  static float VOLATILITY = 0.30f;
	float RandFloat(float low, float high);
	int numOptions;
	int optionSize;


public:
	ElasticBSPricer();
	ElasticBSPricer(LaunchParameters &launchConfig, std::string name, int numOptions);

	void initKernel();
	void runKernel(cudaStream_t &streamToRunIn);
	cudaFuncAttributes getKernelProperties();
	void freeResources();
	size_t getMemoryConsumption();

	virtual ~ElasticBSPricer();

};

#endif /* ELASTICBSPRICER_H_ */
