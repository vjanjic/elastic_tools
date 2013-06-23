/**
 * ElasticDummy.cpp
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ElasticDummy.h"
//#include "cudaCode/dummy_elastic.cu"


ElasticDummy::ElasticDummy(gridParams_logical gridParL, blockParams_logical blockParL) :
		AbstractElasticKernel(gridParL, blockParL) {

	// TODO Auto-generated constructor stub

}

ElasticDummy::~ElasticDummy() {
	// TODO Auto-generated destructor stub
}

void ElasticDummy::initKernel() {
}

void ElasticDummy::runKernel() {
	KernelLimits limits;

	limits.sharedMem = 10;
	limits.blocks = 10;
	limits.registers = 100;
	limits.threads = 100;
	lauch_dummy_elastic(this->lBlock, this->lGrid, limits);

}
