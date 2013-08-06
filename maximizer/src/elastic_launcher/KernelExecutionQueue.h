/**
 * KernelExecutionQueue.h
 *
 * This class is an implementation of a non serializing execution queue which purpose is to launch CUDA
 * kernels while avoiding implicit serialization caused by calls to the NVIDIA API. For that reason all
 * allocation of resources and their disposal is done at bulk before and after all kernels are launched.
 * In addition to that each kernel added to the queue is assigned a separate cuda stream to run in in order
 * to promote concurrency. This data structure can hold any number of kernels as long as their total global
 * GPU memory consumption does not exceed the amount available on the particular GPU that is installed on the
 * system. In addition to that the queue supports other methods for optimization such as maximum concurrency
 * promotion with minimum kernel molding.
 *
 *  Created on: Aug 1, 2013
 *
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

	size_t maxGlobalMem; // the maximum global memory of the GPU
	size_t memoryUsed; // the memory used by the kernels that are in the queue

	// a vector that holds pairs of kernels and cuda streams, so each kernel is associated with a stream to run in
	std::vector<std::pair<boost::shared_ptr<AbstractElasticKernel>, cudaStream_t> > kernelsAndStreams;

	/**
	 * Returns the total free memory on the GPU
	 * @return the free memory on the GPU
	 */
	size_t getFreeGPUMemory();

	/**
	 * Method used in the recursive generation of unique kernel combinations of length N from a set of kernels.
	 * This methods creates all possible combinations and produces a matrix that contains them along with the
	 * calculated scale index of any combination. This scale index indicated how much the kernels in the combination
	 * need to be scaled down (by tweaking their launch parameters) in order to run concurrently. From that matrix,
	 * the set of unique combinations that need the least tweaking is chosen and those are ordered into the queue.
	 * This promotes maximum concurrency of kernels with minimum launch paramter modification
	 *
	 * @param offset where to start from in the set
	 * @param k the size of every combination
	 * @param combination the vector to put the combination in
	 * @param elems  the elements
	 * @param results the results
	 */
	void generateKernelCombinations(int offset, int k, std::vector<boost::shared_ptr<AbstractElasticKernel> > &combination,
			std::vector<boost::shared_ptr<AbstractElasticKernel> >& elems,
			std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> >& results);

	/**
	 * From a vector of kernel combinations and relative scaling factor, this method extracts, the
	 * unique combinations by first sorting the combination vector in increasing order by the parameter
	 * modification index contained in each combination. Since this index indicates how much the configuration
	 * of kernels within a combination need to change in order for them to run concurrently, the ones that end up
	 * in the beginning of the sorted list are the ones needing the least parameter modification and therefore are the
	 * chosen ones.
	 *
	 * @param combinations the combination matrix
	 * @return a vector of extracted kernels that can be added back to the queue
	 */
	std::vector<boost::shared_ptr<AbstractElasticKernel> > extractKernelSequenceWithMinModification(
			std::vector<std::pair<std::vector<boost::shared_ptr<AbstractElasticKernel> >, double> > combinations);

	/**
	 * Given a kernel and a vector, the method finds out whether the kernel is contained in the vector
	 *
	 * @param kernelVector a vector of kernel pointers
	 * @param vector a pointer to a kernel
	 * @return
	 */
	bool isKernelInVector(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector, boost::shared_ptr<AbstractElasticKernel> kernel);

	/**
	 * Given two vectors of kernel pointers, the method finds out whether they contain a common element
	 *
	 * @param kernelVector_x the first vector
	 * @param kernelVector_y the second vector
	 *
	 * @return bool
	 */
	bool doVectorsContainCommonElem(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_x,
			std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelVector_y);
	/**
	 * Given a vector of kernels, limit their resource usage in order to promote concurrency between them.
	 *
	 * @param kernels vector of shared pointers to kernels
	 */

	void limitKernels(std::vector<boost::shared_ptr<AbstractElasticKernel> > kernels);
public:
	/**
	 * Default constructor
	 */
	KernelExecutionQueue();
	/**
	 * Default destructor
	 */
	virtual ~KernelExecutionQueue();

	/**
	 * Adds a kernel to the queue. If the kernel cannot fit into the queue (due to memory restriction) a false bool is returned
	 *
	 * @param kernel the kernel to be added
	 * @return true or false depending on whether kernel fits in the queue
	 */
	bool addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);

	/**
	 * Initializes all the kernels within the queue. This is usually allocating memory on the card
	 */
	void initKernels();

	/**
	 * Modifies the order of the kernels within the queue in order to promote concurrency
	 * between two kernels. This method ensures that the maximum level of concurrency is
	 * achieved with the minimum amount of modification to launch parameters of the kernels
	 */
	void moldQueueForMaximumConcurency();

	/**
	 * Runs all the kernels within the queue
	 */
	void runKernels();

	/**
	 * Frees all the resources that were allocated by the kernels when the init() method was called
	 */
	void disposeQueue();

	/**
	 * Returns the average compute utilization for this queue
	 *
	 * @return double
	 */

	double getComputeOccupancyForQueue();
	/**
	 * Returns the average storage utilization for this queue
	 *
	 * @return double
	 */
	double getStorageOccupancyForQueue();

	// jsut a friend for enable pumping the object into an ostream
	friend std::ostream &operator<<(std::ostream &output, const KernelExecutionQueue &q);
}
;

#endif /* KERNELEXECUTIONQUEUE_H_ */
