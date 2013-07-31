/**
 * starter.cpp
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */
#include "Chunking_elastic/elastic_wrapper/ElasticChunker.h"
#include "BlackScholes_elastic/elastic_wrapper/ElasticBSPricer.h"
#include  "ScalarProduct_elastic/elastic_wrapper/ElasticScalarProduct.h"
#include  "VectorAddition_elastic/elastic_wrapper/ElasticVectorAddition.h"
#include  "elastic_launcher/ElasticKernelMaker.h"
#include "elastic_launcher/KernelScheduler.h"
#include "stdio.h"
#include "cuda_runtime.h"



int main() {

	KernelScheduler scheduler;
	scheduler.addKernel(makeElasticKernel(128, 32, CHUNKING, "chunking"));
	scheduler.addKernel(makeElasticKernel(128, 32, VECTOR_ADD, "vector_addition"));
	scheduler.addKernel(makeElasticKernel(128, 32, BLACK_SCHOLES, "black_scholes"));
	scheduler.addKernel(makeElasticKernel(128, 128, SCALAR_PRODUCT, "scalar_product"));
	scheduler.addKernel(makeElasticKernel(128, 64, MATRIX_MULT, "matrix_multiplication"));
	scheduler.runKernels();


}

