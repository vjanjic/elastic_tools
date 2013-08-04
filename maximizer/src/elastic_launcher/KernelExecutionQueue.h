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
#include "../misc/Macros.h"
#include "../elastic_kernel/occupancy_tools/OccupancyCalculator.h"
#include <cmath>


struct ConcurencyVectorComparator {
	inline bool operator()(std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> i,
			std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> j) {
		return (i.second < j.second);
	}
};

class KernelExecutionQueue {

private:
	static const size_t MEM_ACCURACY = 4;
	static const size_t MEM_SAFEGUARD = 100;

	size_t maxGlobalMem;
	size_t memoryUsed;
	std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> > kernelsAndStreams;
	size_t getFreeGPUMemory(size_t accuracy, size_t safeGuardAmount);
	void generateKernelCombinations(int offset, int k, std::vector<boost::shared_ptr<AbstractElasticKernel> > &combination,
			std::vector<boost::shared_ptr<AbstractElasticKernel> >& elems,
			std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >& results);
	bool isKernelInVector(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector, boost::shared_ptr<AbstractElasticKernel> vector);
	bool doVectorsContainCommonElem(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_x,
			std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_y);
	std::vector<boost::shared_ptr<AbstractElasticKernel> > extractKernelSequenceWithMinModification(std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> > combinations);
	void limitKernels(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernels);
public:
	KernelExecutionQueue();
	virtual ~KernelExecutionQueue();
	bool addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);
	void initKernels();
	void combineKernel();
	void runKernels();
	void disposeQueue();
	friend std::ostream &operator<<(std::ostream &output, const KernelExecutionQueue &q);
}
;

#endif /* KERNELEXECUTIONQUEUE_H_ */
