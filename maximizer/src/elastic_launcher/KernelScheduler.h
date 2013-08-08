/**
 * KernelScheduler.h
 *
 * An instance of this class is a software based kernel scheduler for CUDA kernels.
 * The scheduler can execute a group of kernels of arbitrary size as long as they
 * extend the AbstractElasticKernel class that is provided. The scheduler takes care
 * of organizing the kernels into Execution queues according to the different
 * optimization policies applied on runtime.
 *
 *  Created on: Jul 30, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELSCHEDULER_H_
#define KERNELSCHEDULER_H_
#pragma once
#include <vector>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include "../abstract_elastic_kernel/AbstractElasticKernel.hpp"
#include "KernelExecutionQueue.h"
#include "../occupancy_tools/OccupancyCalculator.h"
#include "cuda_runtime.h"
#include "../misc/SimpleTimer.h"

/**
 * This is an enum which contains the different policies that can be applied.
 * All the policies but the native one use Kernel Execution queues which i s
 * a data structure that promoted kernel concurrency by launching all kernels
 * that is contains in separate cuda streams
 */
enum OptimizationPolicy {
	/**
	 * The native policy simply executes the kernels into the order of which
	 * they have been enqueued, allocating and freeing GPU resources on each
	 * kernel execution. Moreover, every kernel is launched into the default
	 * CUDA stream, which causes Implicit serialization of launches
	 */
	NATIVE,
	/**
	 * The fair policy takes the optimization one step; further. All the kernels
	 * are organized into a number of execution queues, in way that each queue
	 * holds kernels whose total sum of global GPU memory required is no more
	 * than the memory available on the specific GPU that is being used.
	 *
	 */
	FAIR,
	/**
	 * The fair maximum occupancy policy ensures that the launch configuration of every kernel
	 * in all the queues is optimal for the particular GPU. This is done by calculating the
	 * optimal thread block size for the particular kernel.
	 */
	FAIR_MAXIMUM_OCCUPANCY,
	/**
	 * The minimum queues policy does not mold the kernel execution parameters but applies another
	 * optimization that benefits performance. Since each queue capacity is limited by the total
	 * GPU global memory, this policy applies a first fit decreasing approach to the problem of
	 * fitting all the kernels into as few as possible number of queues. Since inside the queues
	 * there is no implicit serialization points for the kernels, this policy promotes more
	 * kernel concurrency within a single execution queue.
	 */
	MINIMUM_QUEUES,
	/**
	 * This policy combines the minimum queues one with the maximum occupancy in order to ensure
	 * that the kernels occupy the minimum number of queues needed and are separately optimized
	 * for maximum theoretical GPU occupancy.
	 */
	MINIMUM_QUEUES_MAXIMUM_OCCUPANCY,
	/**
	 * The maximum concurrency policy rearranges each queue in order to promote pair wise concurrency
	 * with the minimum amount of kernel modification. This is done by enumerating all kernel execution
	 * combinations within a queue and determining which pairs of elastic kernels need the least amount
	 * of modification to their launch parameters in order to promote concurrent kernel execution
	 */
	MAXIMUM_CONCURENCY

};

/**
 * This data structure is used for containing the measures of compute and storage
 * utilization of the whole kernel scheduling configuration for a particular policy
 */
struct GPUUtilization {
	double averageComputeOccupancy;
	double averageStorageOccupancy;
};

/**
 * This comparator object is used to sort the vector of kernels in decreasing order
 * according to their memory consumption. This is needed in order to apply a 1-D
 * bin packing problem and minimize the number of queues used by the scheduler for
 *  a particular set of kernels to be executed.
 */
struct kernelMemConsumptionComparator {
	inline bool operator()(boost::shared_ptr<AbstractElasticKernel> i, boost::shared_ptr<AbstractElasticKernel> j) {
		return (i.get()->getMemoryConsumption() > j.get()->getMemoryConsumption());
	}
};

class KernelScheduler {
private:
	std::vector<boost::shared_ptr<AbstractElasticKernel> > kernelsToRun; // kernels that are enqueues go here...
	std::vector<KernelExecutionQueue> kernelQueues; // kernel queues go here

	/**
	 * Method is used to sort the added to the scheduler kernels in decreasing order
	 *  according to memory consumption
	 */
	void sortKernelByMemoryConsumption();

	/**
	 * This method is used to molds the kernel launch configuration in a way that promotes
	 * maximum theoretical occupancy for a particular kernel on a particular GPU
	 *
	 * @param kernel a shared pointer to the abstract kernel
	 */
	void moldKernelLaunchConfigForMaximumOccupancy(boost::shared_ptr<AbstractElasticKernel> kernel);

	/**
	 * This method iterates through all the queues and optimizes each one for maximum
	 * concurrency with minimum launch parameter modification
	 */
	void optimiseQueuesForMaximumConcurency();

	/**
	 * This method orders the kernels in execution queues by applying the fair policy
	 *
	 */
	void orderKernelsInQueues_FAIR_();

	/**
	 * This method orders kernels in execution queues by minimizing the number of queues
	 * needed
	 */
	void orderKernelsInQueues_MINIMUM_QUEUES_();

	/**
	 * The method simply iterates through all the enqueued kernels and endures each launch
	 * configuration is optimized for the particular GPU that is being used in the system
	 */
	void moldKernels_MAXIMUM_OCCUPANCY_();

	/**
	 * The method is used to apply an optimization policy to the kernels that are enqueued
	 * into the scheduler
	 * @param policy the particular policy to be applied
	 */
	void applyOptimisationPolicy(OptimizationPolicy policy);

public:

	/**
	 * Just a default constructor for the object
	 */
	KernelScheduler();

	/**
	 * Enqueues kernel into the scheduler (optimization is not applied at this point)
	 *
	 * @param kernel a shared pointer to the kernel
	 */
	void addKernel(boost::shared_ptr<AbstractElasticKernel> kernel);

	/**
	 * This method runs all the kernels that have been enqueued with the specified
	 * concurrency policy, making all the necessary adjustments to the kernels and their
	 * placement in execution queues
	 *
	 * @param policy the optimization policy
	 * @param preferedberOfConcurentKernels in case the policy is the maximum
	 * concurrency one, this parameter specifies what is the number of kernels for
	 * which concurrent optimization should be applied (within a queue)
	 *
	 * @return the time it took to run all the kernels
	 */
	double runKernels(OptimizationPolicy policy, int preferedberOfConcurentKernels = 2);

	/**
	 * This method is sort of a dry run of the policy and is used in performance testing.
	 * Its purpose is to deliver data regarding the storage and theoretical compute occupancy of
	 * the kernels for a particular policy
	 *
	 * @param policy the policy being applied
	 * @return a structure containing the average storage and compute utilization of each queue
	 */
	GPUUtilization getGPUOccupancyForPolicy(OptimizationPolicy policy);

	//Default destructor
	virtual ~KernelScheduler();
	friend std::ostream &operator<<(std::ostream &output, const KernelScheduler &sch);

};

#endif /* KERNELSCHEDULER_H_ */
