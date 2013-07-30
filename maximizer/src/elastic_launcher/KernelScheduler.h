/**
 * KernelScheduler.h
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELSCHEDULER_H_
#define KERNELSCHEDULER_H_
#include <vector>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include "../elastic_kernel/AbstractElasticKernel.hpp"
#include "cuda_runtime.h"

class KernelScheduler {
private:
	std::vector<std::pair< boost::shared_ptr<AbstractElasticKernel> , cudaStream_t> > kernelsAndStreams;

	void initKernels();
	void freeResources();
public:
	KernelScheduler();
	void addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);
	void runKernels();
	virtual ~KernelScheduler();
};

#endif /* KERNELSCHEDULER_H_ */
