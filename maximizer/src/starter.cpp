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
	scheduler.addKernel(makeElasticKernel(192,5,CHUNKING,"test"));
	scheduler.addKernel(makeElasticKernel(192,5,BLACK_SCHOLES,"test3"));
	scheduler.addKernel(makeElasticKernel(192,10,CHUNKING,"test"));
	scheduler.runKernels();




}

