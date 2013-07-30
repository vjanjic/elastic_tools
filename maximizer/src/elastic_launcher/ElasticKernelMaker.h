/**
 * ElasticKernelMaker.h
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ELASTICKERNELMAKER_H_
#define ELASTICKERNELMAKER_H_

#include "../elastic_kernel/LaunchParameters.hpp"
#include "../elastic_kernel/AbstractElasticKernel.hpp"
#include "../Chunking_elastic/elastic_wrapper/ElasticChunker.h"
#include "../BlackScholes_elastic/elastic_wrapper/ElasticBSPricer.h"
#include "../ScalarProduct_elastic/elastic_wrapper/ElasticScalarProduct.h"
#include "../VectorAddition_elastic/elastic_wrapper/ElasticVectorAddition.h"
#include <boost/shared_ptr.hpp>


#include "string.h"
#include <string>
enum KernelType {
	CHUNKING, BLACK_SCHOLES, SCALAR_PRODUCT, VECTOR_ADD
};

boost::shared_ptr<AbstractElasticKernel> makeElasticKernel(size_t threadsPerBlock, size_t blocksPerGrid, KernelType type, std::string name) {
	LaunchParameters parameters = LaunchParameters(threadsPerBlock, blocksPerGrid);
	AbstractElasticKernel* result;

	if (type == 0) {

		return boost::shared_ptr<AbstractElasticKernel>(new ElasticChunker(parameters, name));
	}
	if (type == 1) {
		return boost::shared_ptr<AbstractElasticKernel>(new ElasticBSPricer(parameters, name));

	}
	if (type == 2) {
		return boost::shared_ptr<AbstractElasticKernel>(new ElasticScalarProduct(parameters, name));

	}
	if (type == 3) {
		return boost::shared_ptr<AbstractElasticKernel>(new ElasticVectorAddition(parameters, name));

	}

}

#endif /* ELASTICKERNELMAKER_H_ */
