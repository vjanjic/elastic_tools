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

void KernelScheduler::runKernels() {
	this->orderKernelsInQueues();
	int queueNum = 1;

	for (std::vector<KernelExecutionQueue>::iterator it = this->kernelQueues.begin(); it != this->kernelQueues.end(); ++it) {
		std::cout << "Running kernel queue [" << queueNum << "]" << std::endl << (*it) << std::endl;
		++queueNum;
		(*it).initKernels();
		(*it).runKernels();
		(*it).disposeQueue();

	}

	/*
	 * we'll think about this one a little bit more over a cup of coffee......... :(
	 */

}

void KernelScheduler::orderKernelsInQueues() {

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

std::ostream& operator <<(std::ostream& output, const KernelScheduler& sch) {

	for (std::vector<KernelExecutionQueue>::const_iterator it = sch.kernelQueues.begin(); it != sch.kernelQueues.end(); ++it) {
		output << (*it) << std::endl;
	}
}
