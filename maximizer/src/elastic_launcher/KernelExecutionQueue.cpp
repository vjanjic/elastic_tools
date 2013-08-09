/**
 * KernelExecutionQueue.cpp
 *
 *  Created on: Aug 1, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "KernelExecutionQueue.h"

size_t KernelExecutionQueue::getFreeGPUMemory() {
	size_t total;
	size_t free;
	// we just use the nvidia API here to get that...
	cudaMemGetInfo(&free, &total);

	return free - (64 * (1 << 20)); // give a little bit of a buffer for paging...
}

void KernelExecutionQueue::generateKernelCombinations(int offset, int k, std::vector<boost::shared_ptr<AbstractElasticKernel> >& combination,
		std::vector<boost::shared_ptr<AbstractElasticKernel> >& elems,
		std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >& results) {

	if (k == 0) {
		// here we calculate how much the kernels need to change in order to run them concurrently
		double changeIndex = 0;
		double resourceFraction = 1 / (double) combination.size(); // Calculate the resource fraction based on the size of the combination
		for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = combination.begin(); it != combination.end(); ++it) {
			// use the kernel limiter algorithm in order to fit the kernel in the resources avaible to it
			int newThrCount =
					limitKernel((*it), KernelLimits(resourceFraction, resourceFraction, resourceFraction, resourceFraction, getGPUConfiguration())).getNumTotalThreads();
			//Calculate how much the kernel changes in terms of threadcount
			double currentThrs = (double) (*it).get()->getLaunchParams().getNumTotalThreads();
			double delta = std::abs((double) (newThrCount - currentThrs) / currentThrs);
			changeIndex = changeIndex + delta;
		}
		//calcualte the average change index and put it in a pair together with the combination
		changeIndex = changeIndex / (double) combination.size();
		results.push_back(std::make_pair(combination, changeIndex));
		return;
	}
	for (int i = offset; i <= elems.size() - k; ++i) {
		combination.push_back(elems[i]);
		generateKernelCombinations(i + 1, k - 1, combination, elems, results);
		combination.pop_back();
	}

}

std::vector<boost::shared_ptr<AbstractElasticKernel> > KernelExecutionQueue::extractKernelSequenceWithMinModification(
		std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> > combinations) {
	ConcurencyVectorComparator comparator; // Create comparator object to sort the combinations according to change index (decreasing order)
	std::sort(combinations.begin(), combinations.end(), comparator);
	std::vector<boost::shared_ptr<AbstractElasticKernel> > results; // create results vector - this is where final kenrel order will be created

	for (std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >::const_iterator it = combinations.begin();
			it != combinations.end(); ++it) {
		// wal thorugh all combination

		if (!this->doVectorsContainCommonElem(results, (*it).first)) {
			// iff non of the kenrels in the combination are already in the result vector, scale then dodown and add them
			this->limitKernels((*it).first);
			results.insert(results.end(), (*it).first.begin(), (*it).first.end());
			continue;
		}

	}
	if (this->kernelsAndStreams.size() != results.size()) {
		// ensurethat no kernel was left behind due to not perfect division by the size of the combinations
		for (std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >::const_iterator it = combinations.begin();
				it != combinations.end(); ++it) {

			for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it2 = (*it).first.begin(); it2 != (*it).first.end(); ++it2) {

				if (!this->isKernelInVector(results, (*it2))) {
					results.push_back((*it2));
				}

			}

		}
	}

	return results;
}

bool KernelExecutionQueue::isKernelInVector(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector,
		boost::shared_ptr<AbstractElasticKernel> vector) {
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it = kernelVector.begin(); it != kernelVector.end(); ++it) {
		if ((*it) == vector) {
			// simply check whether the kernel is in the vector... we can compare shared pointers just like raw ones.. (thanks boost...)
			return true;
		}
	}
	return false;
}

bool KernelExecutionQueue::doVectorsContainCommonElem(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_x,
		std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_y) {

	// this method is a tad expensive.... but what can we do.. ?
	// simply do intersection of vectors
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it = kernelVector_x.begin(); it != kernelVector_x.end(); ++it) {
		if (this->isKernelInVector(kernelVector_y, (*it))) {
			return true;
		}
	}
	return false;
}

void KernelExecutionQueue::limitKernels(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernels) {

	double resoruceFraction = 1.0 / (double) kernels.size(); // calculate limit fraction
	KernelLimits limit = KernelLimits(resoruceFraction, resoruceFraction, resoruceFraction, resoruceFraction, getGPUConfiguration());
	// just iterate through all kernels and limit their occupancy in order to fit in the hardware constraints imposed
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it = kernels.begin(); it != kernels.end(); ++it) {
		LaunchParameters currentParams = (*it).get()->getLaunchParams();
		LaunchParameters newParams = limitKernel((*it), limit);

		(*it).get()->setLaunchParams(newParams);
	}

}

KernelExecutionQueue::KernelExecutionQueue() {

	this->maxGlobalMem = getFreeGPUMemory();  // just grab the free memory on initiation
	this->memoryUsed = 0;
}

KernelExecutionQueue::~KernelExecutionQueue() {
}

bool KernelExecutionQueue::addKernel(boost::shared_ptr<AbstractElasticKernel> kernel) {

	if (kernel.get()->getMemoryConsumption() < (this->maxGlobalMem - this->memoryUsed)) { // check whether we can fit the kernel in here
		cudaStream_t streamForKernel;
		cudaStreamCreate(&streamForKernel);
		this->kernelsAndStreams.push_back(std::make_pair(kernel, streamForKernel)); // associate kernel with a stream
		this->memoryUsed = this->memoryUsed + kernel.get()->getMemoryConsumption(); // modify memory consumtion of the whole queue
		return true;
	}
	return false;
}

void KernelExecutionQueue::initKernels() {
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		(*it).first.get()->initKernel(); // just init all the kernels in the queue
	}

}

void KernelExecutionQueue::runKernels() {
	//this->initKernels();
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		(*it).first.get()->runKernel((*it).second); // run them all in the perticular stream associated with each kernel
	}
}

void KernelExecutionQueue::disposeQueue() {
	// free all resourcs allocated by the kernels
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		(*it).first.get()->freeResources();
		cudaStreamDestroy((*it).second);
	}
	cudaDeviceSynchronize();
	this->kernelsAndStreams.clear();
	this->memoryUsed = 0;
	this->kernelsAndStreams.begin();
}

void KernelExecutionQueue::moldQueueForMaximumConcurency() {
	if (this->kernelsAndStreams.size() > 1) {
		std::vector<boost::shared_ptr<AbstractElasticKernel> > kernels;

		for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
				it != this->kernelsAndStreams.end(); ++it) {
			kernels.push_back((*it).first); // extract all kernels from the {kernel,stream} pairs
		}
		//clear up the queue
		this->kernelsAndStreams.clear();
		this->memoryUsed = 0;
		std::vector<boost::shared_ptr<AbstractElasticKernel> > comb; // the combination vector
		std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> > results; // the combination matrix along with scaling factor
		this->generateKernelCombinations(0, 2, comb, kernels, results); // generate combinations of 2

		std::vector<boost::shared_ptr<AbstractElasticKernel> > extracted = extractKernelSequenceWithMinModification(results); // extract the kernels with least modification and max concurency

		for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator iter = extracted.begin(); iter != extracted.end(); ++iter) {
			// add kernels back to the queue
			this->addKernel((*iter));
		}
	}

}

double KernelExecutionQueue::getComputeOccupancyForQueue() {
	double result = 0;
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::const_iterator it = kernelsAndStreams.begin();
			it != kernelsAndStreams.end(); ++it) {
		// jsut calcualte average theoretical occupancy for the queue (COMPUTE)
		result = result + getOccupancyForKernel((*it).first);
	}
	return result / (double) this->kernelsAndStreams.size();
}

double KernelExecutionQueue::getStorageOccupancyForQueue() {

	// Calculate average storage occupancy for the whole queue
	double result = 0;
	double gpuMem = (double) this->maxGlobalMem;
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::const_iterator it = kernelsAndStreams.begin();
			it != kernelsAndStreams.end(); ++it) {
		result = result + (*it).first.get()->getMemoryConsumption();
	}

	return result / gpuMem;
}

std::ostream& operator <<(std::ostream& output, const KernelExecutionQueue& q) {

	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::const_iterator it = q.kernelsAndStreams.begin();
			it != q.kernelsAndStreams.end(); ++it) {
		output << *((*it).first.get()) << std::endl;
	}

	return output;

}
