/**
 * KernelExecutionQueue.cpp
 *
 *  Created on: Aug 1, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "KernelExecutionQueue.h"

KernelExecutionQueue::KernelExecutionQueue() {

	this->maxGlobalMem = getFreeGPUMemory(MEM_ACCURACY, MEM_SAFEGUARD);
	//std::cout << std::endl << this->maxGlobalMem << std::endl;
	this->memoryUsed = 0;
}

size_t KernelExecutionQueue::getFreeGPUMemory(size_t accuracy, size_t safeGuardAmount) {
	size_t total;
	size_t free;

	cudaMemGetInfo(&free, &total);

	/*	size_t bytesInMB = 1048576;
	 size_t granularity = accuracy;
	 size_t allocated = 0;
	 int *dummy;

	 while (true) {

	 cudaError_t err = cudaMalloc((void **) &dummy, 1048576 * granularity);
	 if (err != cudaSuccess) {
	 return (allocated - safeGuardAmount) * bytesInMB;
	 }
	 printf("%d\n",allocated);
	 allocated = allocated + granularity;
	 }
	 // just so it does not complain for non return
	 return (allocated - safeGuardAmount) * bytesInMB;*/
	//printf("%d\n", free);
	return free - (64 * (1 << 20));
}

KernelExecutionQueue::~KernelExecutionQueue() {
}

bool KernelExecutionQueue::addKernel(boost::shared_ptr<AbstractElasticKernel> kernel) {

	if (kernel.get()->getMemoryConsumption() < (this->maxGlobalMem - this->memoryUsed)) {
		cudaStream_t streamForKernel;
		cudaStreamCreate(&streamForKernel);
		this->kernelsAndStreams.push_back(std::make_pair(kernel, streamForKernel));
		this->memoryUsed = this->memoryUsed + kernel.get()->getMemoryConsumption();
		//printf("%d\n",memoryUsed);
		return true;
	}
	return false;
}

void KernelExecutionQueue::initKernels() {
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		(*it).first.get()->initKernel();
	}

}

void KernelExecutionQueue::runKernels() {
	//this->initKernels();
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		(*it).first.get()->runKernel((*it).second);
	}
}

void KernelExecutionQueue::disposeQueue() {
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

void KernelExecutionQueue::generateKernelCombinations(int offset, int k, std::vector<boost::shared_ptr<AbstractElasticKernel> >& combination,
		std::vector<boost::shared_ptr<AbstractElasticKernel> >& elems,
		std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >& results) {

	if (k == 0) {
		double changeIndex = 0;

		for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::iterator it = combination.begin(); it != combination.end(); ++it) {
			int newThrCount = limitKernel((*it), KernelLimits(0.1, 0.1, 0.1, 0.1, getGPUConfiguration())).getNumTotalThreads();
			double currentThrs = (double) (*it).get()->getLaunchParams().getNumTotalThreads();
			double delta = std::abs((double) (newThrCount - currentThrs) / currentThrs);
			changeIndex = changeIndex + delta;
		}

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

void KernelExecutionQueue::combineKernel() {

	std::vector<boost::shared_ptr<AbstractElasticKernel> > kernels;

	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		kernels.push_back((*it).first);
	}
	this->kernelsAndStreams.clear();
	this->memoryUsed = 0;
	std::vector<boost::shared_ptr<AbstractElasticKernel> > comb;
	std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> > results;
	this->generateKernelCombinations(0, 2, comb, kernels, results);
	for (std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >::iterator it = results.begin(); it != results.end(); ++it) {

		for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it2 = (*it).first.begin(); it2 != (*it).first.end(); ++it2) {
			//std::cout << *(*it2).get() << " | ";
		}
		//std::cout << (*it).second << std::endl;
	}

	std::vector<boost::shared_ptr<AbstractElasticKernel> > extracted = extractKernelSequenceWithMinModification(results);

	//std::cout << "-----------------------------------------------" << std::endl;
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator iter = extracted.begin(); iter != extracted.end(); ++iter) {
		//std::cout << *(*iter).get() << std::endl;
		this->addKernel((*iter));
	}

}

bool KernelExecutionQueue::isKernelInVector(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector,
		boost::shared_ptr<AbstractElasticKernel> vector) {
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it = kernelVector.begin(); it != kernelVector.end(); ++it) {
		if ((*it) == vector) {
			return true;
		}
	}
	return false;
}

bool KernelExecutionQueue::doVectorsContainCommonElem(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_x,
		std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_y) {
	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it = kernelVector_x.begin(); it != kernelVector_x.end(); ++it) {
		if (this->isKernelInVector(kernelVector_y, (*it))) {
			return true;
		}
	}
	return false;
}

std::vector<boost::shared_ptr<AbstractElasticKernel> > KernelExecutionQueue::extractKernelSequenceWithMinModification(
		std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> > combinations) {
	ConcurencyVectorComparator comparator;
	std::sort(combinations.begin(), combinations.end(), comparator);
	std::vector<boost::shared_ptr<AbstractElasticKernel> > results;

	for (std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >::const_iterator it = combinations.begin();
			it != combinations.end(); ++it) {
		if (!this->doVectorsContainCommonElem(results, (*it).first)) {
			this->limitKernels((*it).first);
			results.insert(results.end(), (*it).first.begin(), (*it).first.end());
			continue;
		}

	}
	if (this->kernelsAndStreams.size() != results.size()) {
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

void KernelExecutionQueue::limitKernels(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernels) {

	double resoruceFraction = 1.0 / (double) kernels.size();
	KernelLimits limit = KernelLimits(resoruceFraction, resoruceFraction, resoruceFraction, resoruceFraction, getGPUConfiguration());

	for (std::vector<boost::shared_ptr<AbstractElasticKernel> >::const_iterator it = kernels.begin(); it != kernels.end(); ++it) {
		LaunchParameters currentParams = (*it).get()->getLaunchParams();
		LaunchParameters newParams = limitKernel((*it), limit);

		//std::cout << currentParams << " ---- " << newParams<< std::endl;
		(*it).get()->setLaunchParams(newParams);
	}

}

double KernelExecutionQueue::getComputeOccupancyForQueue() {
	double result = 0;
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::const_iterator it = kernelsAndStreams.begin();
			it != kernelsAndStreams.end(); ++it) {
		result = result + getOccupancyForKernel((*it).first);
	}
	return result / (double) this->kernelsAndStreams.size();
}

double KernelExecutionQueue::getStorageOccupancyForQueue() {


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
