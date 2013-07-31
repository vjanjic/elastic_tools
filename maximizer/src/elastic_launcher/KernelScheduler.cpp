/**
 * KernelScheduler.cpp
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "KernelScheduler.h"

KernelScheduler::KernelScheduler() {
	// TODO Auto-generated constructor stub

}

void KernelScheduler::addKernel(boost::shared_ptr<AbstractElasticKernel> kernel) {
	cudaStream_t streamForKernel;
	cudaStreamCreate(&streamForKernel);
	this->kernelsAndStreams.push_back(std::make_pair(kernel, streamForKernel));

}

void KernelScheduler::runKernels() {
	this->initKernels();
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		std::cout << "running " << *(*it).first.get() << std::endl;
		(*it).first.get()->runKernel((*it).second);
	}

}

void KernelScheduler::initKernels() {
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		(*it).first.get()->initKernel();
	}
}

void KernelScheduler::freeResources() {
	for (std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> >::iterator it = this->kernelsAndStreams.begin();
			it != this->kernelsAndStreams.end(); ++it) {
		(*it).first.get()->freeResources();
		cudaStreamDestroy((*it).second);
	}
}

size_t KernelScheduler::getFreeGPUMemory(size_t accuracy, size_t safeGuardAmount) {
	size_t bytesInMB = 1048576;
	size_t granularity = accuracy;
	size_t allocated = 0;
	int *dummy = NULL;

	while (true) {

		cudaError_t err = cudaMalloc((void **) &dummy, 1048576 * granularity);
		if (err != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(err));
			cudaFree(dummy);
			return allocated;
		}
		allocated = allocated + granularity;
	}
	// just so it does not complain for non return
	return allocated - safeGuardAmount;
}

KernelScheduler::~KernelScheduler() {
	this->freeResources();
}

