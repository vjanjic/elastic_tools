/**
 * ElasticBSPricer.cpp
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ElasticBSPricer.h"

ElasticBSPricer::ElasticBSPricer() :
		AbstractElasticKernel() {

}

ElasticBSPricer::ElasticBSPricer(LaunchParameters &launchConfig, std::string name) :
		AbstractElasticKernel(launchConfig, name) {
}

void ElasticBSPricer::initKernel() {

	float *h_CallResultCPU, *h_PutResultCPU, *h_CallResultGPU, *h_PutResultGPU, *h_StockPrice, *h_OptionStrike, *h_OptionYears;

	h_CallResultCPU = (float *) malloc(OPT_SZ);
	h_PutResultCPU = (float *) malloc(OPT_SZ);
	h_CallResultGPU = (float *) malloc(OPT_SZ);
	h_PutResultGPU = (float *) malloc(OPT_SZ);
	h_StockPrice = (float *) malloc(OPT_SZ);
	h_OptionStrike = (float *) malloc(OPT_SZ);
	h_OptionYears = (float *) malloc(OPT_SZ);

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_CallResult, OPT_SZ));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_PutResult, OPT_SZ));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_StockPrice, OPT_SZ));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_OptionStrike, OPT_SZ));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&d_OptionYears, OPT_SZ));

	srand(5347);

	//Generate options set
	for (int i = 0; i < OPT_N; i++) {
		h_CallResultCPU[i] = 0.0f;
		h_PutResultCPU[i] = -1.0f;
		h_StockPrice[i] = RandFloat(5.0f, 30.0f);
		h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
		h_OptionYears[i] = RandFloat(0.25f, 10.0f);
	}

	CUDA_CHECK_RETURN(cudaMemcpy(d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_OptionStrike, h_OptionStrike, OPT_SZ, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_OptionYears, h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice));

	free(h_OptionYears);
	free(h_OptionStrike);
	free(h_StockPrice);
	free(h_PutResultGPU);
	free(h_CallResultGPU);
	free(h_PutResultCPU);
	free(h_CallResultCPU);

}

float ElasticBSPricer::RandFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + t * high;
}

void ElasticBSPricer::runKernel(cudaStream_t &streamToRunIn) {

	startBSKernel(gridConfig.getThreadsPerBlock(), gridConfig.getBlocksPerGrid(), d_CallResult, d_PutResult, d_StockPrice, d_OptionStrike, d_OptionYears,
			RISKFREE, VOLATILITY, OPT_N, streamToRunIn);

}

cudaFuncAttributes ElasticBSPricer::getKernelProperties() {
	return getBSKernelProperties();
}

void ElasticBSPricer::freeResources() {

	CUDA_CHECK_RETURN(cudaFree(d_OptionYears));
	CUDA_CHECK_RETURN(cudaFree(d_OptionStrike));
	CUDA_CHECK_RETURN(cudaFree(d_StockPrice));
	CUDA_CHECK_RETURN(cudaFree(d_PutResult));
	CUDA_CHECK_RETURN(cudaFree(d_CallResult));
}

ElasticBSPricer::~ElasticBSPricer() {
	// TODO Auto-generated destructor stub
}

