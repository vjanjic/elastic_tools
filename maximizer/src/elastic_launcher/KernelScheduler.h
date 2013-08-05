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
#include "../misc/SimpleTimer.h"

/*enum OptimizationPolicy {
	FAIR, MINIMUM_QUEUES, MAXIMUM_OCCUPANCY, MAXIMUM_CONCURENCY, NATIVE
};*/

enum OptimizationPolicy {
	NATIVE, FAIR, FAIR_MAXIMUM_OCCUPANCY, MINIMUM_QUEUES, MINIMUM_QUEUES_MAXIMUM_OCCUPANCY, MAXIMUM_CONCURENCY

};

struct GPUUtilization {
	double averageComputeOccupancy;
	double averageStorageOccupancy;

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
	void optimiseQueuesForMaximumConcurency();
	void applyOptimisationPolicy(OptimizationPolicy policy);

public:
	void printOptimisation();
	KernelScheduler();
	void addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);
	double runKernels(OptimizationPolicy policy, int preferedberOfConcurentKernels = 2);
	void orderKernelsInQueues_FAIR_();
	void orderKernelsInQueues_MINIMUM_QUEUES_();
	void moldKernels_MAXIMUM_OCCUPANCY_();
	GPUUtilization getGPUOccupancyForPolicy(OptimizationPolicy policy);

	virtual ~KernelScheduler();

	friend std::ostream &operator<<(std::ostream &output, const KernelScheduler &sch);

};

#endif /* KERNELSCHEDULER_H_ */
