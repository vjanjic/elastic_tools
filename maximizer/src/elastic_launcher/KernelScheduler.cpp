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

	if (policy != 4) {
		if (policy == 0) {
			this->orderKernelsInQueues_FAIR_();
		}
		if (policy == 1) {
			this->orderKernelsInQueues_MINIMUM_QUEUES_();
		}

		if (policy == 2 || policy == 3) {
			this->moldKernels_MAXIMUM_OCCUPANCY_();
			this->orderKernelsInQueues_MINIMUM_QUEUES_();

		}
		if (policy == 3) {
			this->optimiseQueuesForMaximumConcurency();
		}

		//now we run the execution queues
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

std::ostream& operator <<(std::ostream& output, const KernelScheduler& sch) {
	int qNum = 1;
	for (std::vector<KernelExecutionQueue>::const_iterator it = sch.kernelQueues.begin(); it != sch.kernelQueues.end(); ++it) {
		output << "Queue: " << qNum << std::endl;
		output << (*it) << std::endl;
		++qNum;
	}
	return output;
}
