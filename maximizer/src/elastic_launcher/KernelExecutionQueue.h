/**
 * KernelExecutionQueue.h
 *
 *  Created on: Aug 1, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELEXECUTIONQUEUE_H_
#define KERNELEXECUTIONQUEUE_H_

#include <vector>
#include <boost/shared_ptr.hpp>
#include "../elastic_kernel/AbstractElasticKernel.hpp"

class KernelExecutionQueue {

private:
	static const size_t MEM_ACCURACY = 4;
	static const size_t MEM_SAFEGUARD = 100;

	size_t maxGlobalMem;
	size_t memoryUsed;
	std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> > kernelsAndStreams;
	size_t getFreeGPUMemory(size_t accuracy, size_t safeGuardAmount);

public:
	KernelExecutionQueue();
	virtual ~KernelExecutionQueue();
	bool addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);
	void initKernels();

	void runKernels();
	void disposeQueue();
	friend std::ostream &operator<< (std::ostream &output, const KernelExecutionQueue &q);
};

#endif /* KERNELEXECUTIONQUEUE_H_ */
