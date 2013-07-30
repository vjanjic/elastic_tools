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

int main() {



	KernelScheduler scheduler;
	scheduler.addKernel(makeElasticKernel(192,128,CHUNKING,"chunking"));
	scheduler.addKernel(makeElasticKernel(192,1,BLACK_SCHOLES,"black_scholes"));

	scheduler.addKernel(makeElasticKernel(192,5,VECTOR_ADD,"vector_addition"));
	scheduler.addKernel(makeElasticKernel(192,5,SCALAR_PRODUCT,"scalar_product"));
	scheduler.addKernel(makeElasticKernel(32,16,MATRIX_MULT,"matrix_multiplication"));

	scheduler.runKernels();




}

