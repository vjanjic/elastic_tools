/**
 * KernelScheduler.cpp
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "KernelScheduler.h"

KernelScheduler::KernelScheduler() {

}

void KernelScheduler::addKernel(boost::shared_ptr<AbstractElasticKernel> kernel) {
	this->kernelsToRun.push_back(kernel);

}

double KernelScheduler::runKernels(OptimizationPolicy policy, int preferedNumberOfConcurentKernels) {
	SimpleTimer t("debugTimer");

	if (policy != 0) {
		this->applyOptimisationPolicy(policy);
		t.start();
		int queueNum = 1;
		for (std::vector<KernelExecutionQueue>::iterator it = this->kernelQueues.begin(); it != this->kernelQueues.end(); ++it) {
			//std::cout << "Running kernel queue [" << queueNum << "]" << std::endl << (*it) << std::endl;
			++queueNum;
			(*it).initKernels();
			(*it).runKernels();
			(*it).disposeQueue();

		}
		return t.stop();

	} else {
		t.start();
		// if the policy is native.. just serialize all the kernels because that would have happened in real life...
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {

			(*it).get()->initKernel();
			(*it).get()->runKernel(stream);
			(*it).get()->freeResources();

		}
		cudaStreamDestroy(stream);
		return t.stop();
	}

}

void KernelScheduler::orderKernelsInQueues_MINIMUM_QUEUES_() {
	this->sortKernelByMemoryConsumption();
	this->kernelQueues.push_back(KernelExecutionQueue());
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {

		for (std::vector<KernelExecutionQueue>::iterator it2 = this->kernelQueues.begin(); it2 != this->kernelQueues.end(); ++it2) {

			if ((*it2).addKernel(*it)) {
				break;
			}
			if (this->kernelQueues.end() - it2 == 1) {
				KernelExecutionQueue queue;
				queue.addKernel(*it);
				this->kernelQueues.push_back(queue);
				break;
			}

		}

	}

}

void KernelScheduler::orderKernelsInQueues_FAIR_() {
	this->kernelQueues.push_back(KernelExecutionQueue());
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {

		if (!(this->kernelQueues.back().addKernel(*it))) {
			this->kernelQueues.push_back(KernelExecutionQueue());
			--it;
		}

	}

}

KernelScheduler::~KernelScheduler() {

}

void KernelScheduler::sortKernelByMemoryConsumption() {
	kernelMemConsumptionComparator comparator;
	std::sort(this->kernelsToRun.begin(), this->kernelsToRun.end(), comparator);

}

void KernelScheduler::printOptimisation() {
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {

		std::cout << *((*it).get()) << " " << getOptimalBlockSize((*it)) << std::endl;

	}
}

void KernelScheduler::moldKernels_MAXIMUM_OCCUPANCY_() {

	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {

		//std::cout << *(*it).get() << " --------- ";
		moldKernelLaunchConfig((*it));
		//std::cout << *(*it).get() << std::endl;

	}

}

void KernelScheduler::moldKernelLaunchConfig(boost::shared_ptr<AbstractElasticKernel> kernel) {
	size_t newBlockSize = getOptimalBlockSize(kernel);
	LaunchParameters parameters = kernel.get()->getLaunchParams();
	size_t totalThreads = parameters.getNumTotalThreads();

	size_t newBlockNum = totalThreads / newBlockSize;
	if (totalThreads % newBlockSize) {
		++newBlockNum;
	}

	kernel.get()->setLaunchlParams(LaunchParameters(newBlockSize, newBlockNum));

}

void KernelScheduler::optimiseQueuesForMaximumConcurency() {
	for (std::vector<KernelExecutionQueue>::iterator it = this->kernelQueues.begin(); it != this->kernelQueues.end(); ++it) {
		(*it).combineKernel();
	}

}

void KernelScheduler::applyOptimisationPolicy(OptimizationPolicy policy) {

	if (policy == 1) {
		//FAIR
		this->orderKernelsInQueues_FAIR_();
	}
	if (policy == 2) {
		//FAIR maximum occupancy
		this->moldKernels_MAXIMUM_OCCUPANCY_();
		this->orderKernelsInQueues_FAIR_();

	}
	if (policy == 3) {
		this->orderKernelsInQueues_MINIMUM_QUEUES_();

	}
	if (policy == 4) {
		this->moldKernels_MAXIMUM_OCCUPANCY_();
		this->orderKernelsInQueues_MINIMUM_QUEUES_();
	}

	if (policy == 5) {
		this->moldKernels_MAXIMUM_OCCUPANCY_();
		this->orderKernelsInQueues_MINIMUM_QUEUES_();
		this->optimiseQueuesForMaximumConcurency();

	}

}

GPUUtilization KernelScheduler::getGPUOccupancyForPolicy(OptimizationPolicy policy) {
	GPUUtilization result;
	result.averageComputeOccupancy = 0;
	result.averageStorageOccupancy = 0;

	if (policy == 0) {
		for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {
			result.averageStorageOccupancy += getMemoryOccupancyForKernel((*it));
			result.averageComputeOccupancy += getOccupancyForKernel((*it));
		}

		result.averageComputeOccupancy = result.averageComputeOccupancy / (double) this->kernelsToRun.size();
		result.averageStorageOccupancy = result.averageStorageOccupancy / (double) this->kernelsToRun.size();
		return result;
	} else {

		this->applyOptimisationPolicy(policy);

		for (std::vector<KernelExecutionQueue>::iterator it = this->kernelQueues.begin(); it != this->kernelQueues.end(); ++it) {
			result.averageStorageOccupancy += (*it).getStorageOccupancyForQueue();
			result.averageComputeOccupancy += (*it).getComputeOccupancyForQueue();
		}

		result.averageComputeOccupancy = result.averageComputeOccupancy / (double) this->kernelQueues.size();
		result.averageStorageOccupancy = result.averageStorageOccupancy / (double) this->kernelQueues.size();

		return result;

	}

}

std::ostream& operator <<(std::ostream& output, const KernelScheduler& sch) {
	int qNum = 1;
	for (std::vector<KernelExecutionQueue>::const_iterator it = sch.kernelQueues.begin(); it != sch.kernelQueues.end(); ++it) {
		output << "Queue: " << qNum << std::endl;
		output << (*it) << std::endl;
		++qNum;
	}
	return output;
}
