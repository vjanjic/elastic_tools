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
#include "KernelExecutionQueue.h"

#include "cuda_runtime.h"

class KernelScheduler {
private:
	std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelsToRun;
	std::vector<KernelExecutionQueue> kernelQueues;


public:
	KernelScheduler();
	void addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);
	void runKernels();
	void orderKernelsInQueues();


	virtual ~KernelScheduler();

	friend std::ostream &operator<< (std::ostream &output, const KernelScheduler &sch);

};

#endif /* KERNELSCHEDULER_H_ */
