/**
 * KernelScheduler.cpp
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "KernelScheduler.h"

void KernelScheduler::sortKernelByMemoryConsumption() {
	// just using the comparator object to sort them in decreasing order
	kernelMemConsumptionComparator comparator;
	std::sort(this->kernelsToRun.begin(), this->kernelsToRun.end(), comparator);

}

void KernelScheduler::moldKernelLaunchConfigForMaximumOccupancy(boost::shared_ptr<AbstractElasticKernel> kernel) {

	size_t newBlockSize = getOptimalBlockSize(kernel); // get the optimal block size

	LaunchParameters parameters = kernel.get()->getLaunchParams();

	size_t totalThreads = parameters.getNumTotalThreads();
	// make sure we ahve the same amount of threads overall, by decreasing the block size
	size_t newBlockNum = totalThreads / newBlockSize;
	if (totalThreads % newBlockSize) {
		++newBlockNum;
	}
	//set the new configuration
	kernel.get()->setLaunchParams(LaunchParameters(newBlockSize, newBlockNum));

}

void KernelScheduler::optimiseQueuesForMaximumConcurency() {
	// iterate though all the created queues and call their member method for molding or maximum concurency.
	// All the details around that are handled within the execution queue itself
	for (std::vector<KernelExecutionQueue>::iterator it = this->kernelQueues.begin(); it != this->kernelQueues.end(); ++it) {

		(*it).moldQueueForMaximumConcurency();
	}

}

void KernelScheduler::orderKernelsInQueues_FAIR_() {
	this->kernelQueues.push_back(KernelExecutionQueue()); // create a new queue

	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {
		// iterate through all the kernels
		//std::cout << "HO HO HO " << std::endl;;

		if (!(this->kernelQueues.back().addKernel(*it))) {
			// try and add the kernel to the last queue;
			// if it does not fit, simply create a new queue
			this->kernelQueues.push_back(KernelExecutionQueue());
			--it;

		}

	}

}

/**
 * This really is just a first fit decreasing algorithm for optimal one dimensional bin packing
 */
void KernelScheduler::orderKernelsInQueues_MINIMUM_QUEUES_() {
	this->sortKernelByMemoryConsumption(); // sort kernels in decreasing order
	this->kernelQueues.push_back(KernelExecutionQueue()); // create a new queue
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {
		// iterate through all the kernels
		for (std::vector<KernelExecutionQueue>::iterator it2 = this->kernelQueues.begin(); it2 != this->kernelQueues.end(); ++it2) {
			//try and insert kernel into all the queues starting from the beginning
			// if it does not fit, try the next one
			if ((*it2).addKernel(*it)) {
				break;
			}
			if (this->kernelQueues.end() - it2 == 1) {
				// if this is the last kernel queue and it does not fit, add another queue
				KernelExecutionQueue queue;
				queue.addKernel(*it);
				this->kernelQueues.push_back(queue);
				break;
			}

		}

	}

}

void KernelScheduler::moldKernels_MAXIMUM_OCCUPANCY_() {
	// iterate through all the kernels and apply launch parameter transformation in order to promote maximum occupancy
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {
		moldKernelLaunchConfigForMaximumOccupancy((*it));
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
		// minimum queues
		this->orderKernelsInQueues_MINIMUM_QUEUES_();

	}
	if (policy == 4) {
		// both min queue and max occupancy
		this->moldKernels_MAXIMUM_OCCUPANCY_();
		this->orderKernelsInQueues_MINIMUM_QUEUES_();
	}

	if (policy == 5) {


		// all the above + concurency optimization
		this->moldKernels_MAXIMUM_OCCUPANCY_();

		this->orderKernelsInQueues_MINIMUM_QUEUES_();
		this->optimiseQueuesForMaximumConcurency();

	}

}

// we do not need anything in the default constructor
KernelScheduler::KernelScheduler() {

}

void KernelScheduler::addKernel(boost::shared_ptr<AbstractElasticKernel> kernel) {
	// just push back the kernel into the kernel vector
	this->kernelsToRun.push_back(kernel);

}

double KernelScheduler::runKernels(OptimizationPolicy policy, int preferedNumberOfConcurentKernels) {
	SimpleTimer t("debugTimer"); // Create a timer

	if (policy != 0) {

		// apply optmisation strategy (if not native)
		this->applyOptimisationPolicy(policy);
		t.start();
		int queueNum = 1;
		for (std::vector<KernelExecutionQueue>::iterator it = this->kernelQueues.begin(); it != this->kernelQueues.end(); ++it) {
			/*
			 * walk through all the queues and run them
			 */
			++queueNum;
			(*it).initKernels();
			(*it).runKernels();
			(*it).disposeQueue();

		}
		return t.stop(); // stop timer

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

KernelScheduler::~KernelScheduler() {
	//we do not need anything here (the magic of shared pointers....)
}

GPUUtilization KernelScheduler::getGPUOccupancyForPolicy(OptimizationPolicy policy) {

	GPUUtilization result;
	result.averageComputeOccupancy = 0;
	result.averageStorageOccupancy = 0;
	if (policy == 0) {
		//if native policy
		for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = this->kernelsToRun.begin(); it != this->kernelsToRun.end(); ++it) {
			result.averageStorageOccupancy += getMemoryOccupancyForKernel((*it));
			result.averageComputeOccupancy += getOccupancyForKernel((*it));
		}
		//compute statistics for every kernel
		result.averageComputeOccupancy = result.averageComputeOccupancy / (double) this->kernelsToRun.size();
		result.averageStorageOccupancy = result.averageStorageOccupancy / (double) this->kernelsToRun.size();
		return result;
	} else {

		// if not compute for every queue
		this->applyOptimisationPolicy(policy);

		for (std::vector<KernelExecutionQueue>::iterator it = this->kernelQueues.begin(); it != this->kernelQueues.end(); ++it) {
			result.averageStorageOccupancy += (*it).getStorageOccupancyForQueue();
			result.averageComputeOccupancy += (*it).getComputeOccupancyForQueue();
		}

		result.averageComputeOccupancy = result.averageComputeOccupancy / (double) this->kernelQueues.size();
		result.averageStorageOccupancy = result.averageStorageOccupancy / (double) this->kernelQueues.size();

		return result; // return the struct with the information

	}

}

std::ostream& operator <<(std::ostream& output, const KernelScheduler& sch) {
	int qNum = 1;
	// when given to the oastream, this object shoud print all the queues and their content
	for (std::vector<KernelExecutionQueue>::const_iterator it = sch.kernelQueues.begin(); it != sch.kernelQueues.end(); ++it) {
		output << "Queue: " << qNum << std::endl;
		output << (*it) << std::endl;
		++qNum;
	}
	return output;
}
