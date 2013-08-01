/**
 * ElasticScalarProduct.h
 *
 *  Created on: Jul 29, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */


#ifndef ELASTICSCALARPRODUCT_H_
#define ELASTICSCALARPRODUCT_H_
#include "../../elastic_kernel/AbstractElasticKernel.hpp"
#include "../../misc/Macros.h"
#include "../GPU_code/Kernel_Starter_SP.h"
class ElasticScalarProduct: public AbstractElasticKernel {

private:

	//Total number of input vector pairs; arbitrary
	const static int VECTOR_N = 256;
	//Number of elements per vector; arbitrary,
	//but strongly preferred to be a multiple of warp size
	//to meet memory coalescing constraints
	const static int ELEMENT_N = 16384;
	//Total number of data elements
	const static int DATA_N = VECTOR_N * ELEMENT_N;

	const static int DATA_SZ = DATA_N * sizeof(float);
	const static int RESULT_SZ = VECTOR_N * sizeof(float);

	float *h_A;
	float *h_B;
	float *h_C_CPU;
	float *h_C_GPU;
	float *d_A;
	float *d_B;
	float *d_C;
	double delta;
	double ref;
	double sum_delta;
	double sum_ref;
	double L1norm;

	float RandFloat(float low, float high);

public:
	ElasticScalarProduct();
	ElasticScalarProduct(LaunchParameters &launchConfig, std::string name);

	void initKernel();
	void runKernel(cudaStream_t &streamToRunIn);
	cudaFuncAttributes getKernelProperties();
	void freeResources();
	size_t getMemoryConsumption();
	virtual ~ElasticScalarProduct();
};

#endif /* ELASTICSCALARPRODUCT_H_ */
