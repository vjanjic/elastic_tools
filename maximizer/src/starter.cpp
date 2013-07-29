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

#include "stdio.h"

int main() {

	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStream_t stream4;
	cudaStream_t stream5;
	cudaStream_t stream6;


	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	cudaStreamCreate(&stream5);
	cudaStreamCreate(&stream6);



	LaunchParameters parameters = LaunchParameters(128, 3);
	LaunchParameters parameters2 = LaunchParameters(32, 1);

	ElasticChunker chunker1 = ElasticChunker(parameters, "chunker1");
	ElasticChunker chunker2 = ElasticChunker(parameters, "chunker2");
	ElasticChunker chunker3 = ElasticChunker(parameters, "chunker3");
	ElasticBSPricer pricer1 = ElasticBSPricer(parameters, "optionPricer");
	ElasticScalarProduct scalarProduct1 =  ElasticScalarProduct(parameters,"elasticScalarProduct");
	ElasticVectorAddition vectorAdd1 = ElasticVectorAddition(parameters2,"elasticVectorAddition");
	chunker1.initKernel();
	chunker2.initKernel();
	chunker3.initKernel();
	pricer1.initKernel();
	scalarProduct1.initKernel();
	vectorAdd1.initKernel();

	std::cout << "running " << chunker1 << std::endl;
	chunker1.runKernel(stream1);
	std::cout << "running " << chunker2 << std::endl;
	chunker2.runKernel(stream2);
	std::cout << "running " << chunker3 << std::endl;
	chunker3.runKernel(stream3);
	std::cout << "running " << pricer1 << std::endl;
	pricer1.runKernel(stream4);
	std::cout << "running " << scalarProduct1 << std::endl;
	scalarProduct1.runKernel(stream5);
	std::cout << "running " << vectorAdd1 << std::endl;
	vectorAdd1.runKernel(stream6);


	chunker1.freeResources();
	chunker2.freeResources();
	chunker3.freeResources();
	pricer1.freeResources();
	scalarProduct1.freeResources();
	vectorAdd1.freeResources();

}

