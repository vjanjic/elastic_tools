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
	return free - (32*(1 << 20));
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

std::ostream& operator <<(std::ostream& output, const KernelExecutionQueue& q) {

	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::const_iterator it = q.kernelsAndStreams.begin();
			it != q.kernelsAndStreams.end(); ++it) {
		output << *((*it).first.get()) << std::endl;
	}

	return output;

}
