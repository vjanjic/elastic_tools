/**
 * KernelScheduler.h
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELSCHEDULER_H_
#define KERNELSCHEDULER_H_
#pragma once
#include <vector>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include "../elastic_kernel/AbstractElasticKernel.hpp"
#include "KernelExecutionQueue.h"
#include "../elastic_kernel/occupancy_tools/OccupancyCalculator.h"
#include "cuda_runtime.h"

enum OptimizationPolicy {
	FAIR, MINIMUM_QUEUES, MAXIMUM_OCCUPANCY, MAXIMUM_CONCURENCY
};

struct kernelMemConsumptionComparator {
	inline bool operator()(boost::shared_ptr<AbstractElasticKernel> i, boost::shared_ptr<AbstractElasticKernel> j) {
		return (i.get()->getMemoryConsumption() > j.get()->getMemoryConsumption());
	}
};

class KernelScheduler {
private:
	std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelsToRun;
	std::vector<KernelExecutionQueue> kernelQueues;
	void sortKernelByMemoryConsumption();
	void moldKernelLaunchConfig(boost::shared_ptr<AbstractElasticKernel> kernel);

public:
	void printOptimisation();
	KernelScheduler();
	void addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);
	void runKernels(OptimizationPolicy policy);
	void orderKernelsInQueues_FAIR_();
	void orderKernelsInQueues_MINIMUM_QUEUES_();
	void moldKernels_MAXIMUM_OCCUPANCY_();

	virtual ~KernelScheduler();

	friend std::ostream &operator<<(std::ostream &output, const KernelScheduler &sch);

};

#endif /* KERNELSCHEDULER_H_ */
