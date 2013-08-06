/**
 * AbstractElasticKernel.hpp
 *
 * This is an abstract class which can be inherited by implementations of elastic kernels.
 * Elastic kernels are those which can scale up and down by changing their thread and block
 * configuration. This class provides the skeleton for all that functionality. Kernels that
 * need to exhibit elasticity should extend it and implement the functionality in order to do so.
 *
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ABSTRACTELASTICKERNEL_HPP_
#define ABSTRACTELASTICKERNEL_HPP_

#include  "../elastic_kernel/occupancy_tools/OccupancyData.h"
#include "LaunchParameters.hpp"
#include "stdint.h"
#include <iostream>
#include <string.h>
class AbstractElasticKernel {

protected:
	LaunchParameters gridConfig; // the configuration parameters
	std::string name; // the name of the kernel
	size_t memConsumption; // the total global memory consumption of the kernel

public:
	AbstractElasticKernel();
	/**
	 * This constructor takes in a name and a default launch configuration for this kernel
	 *
	 * @param gridConfig the launch configuration of the kernel
	 * @param name the name of the kernel
	 */
	AbstractElasticKernel(const LaunchParameters& gridConfig, std::string name);
	virtual ~AbstractElasticKernel();

	/**
	 * This method takes care of allocating all the resources needed for the kernel to run
	 */
	virtual void initKernel() = 0;

	/**
	 * This method executes the kernel in the specified stream with the configuration
	 *  that was assigned to the kernel
	 *
	 * @param streamToRunIn a cuda stream for the kernel to run in
	 */
	virtual void runKernel(cudaStream_t &streamToRunIn) = 0;

	/**
	 *  This method return the function attributes of this kernel. This is used
	 *  to determine optimal block configuration, etc.
	 * @return
	 */
	virtual cudaFuncAttributes getKernelProperties() = 0;

	/**
	 * The method frees all the resources that were allocated by the kernel
	 */
	virtual void freeResources() = 0;

	/**
	 * The method return the memory consumption of this kernel in bytes
	 * @return
	 */
	virtual size_t getMemoryConsumption() = 0;

	/**
	 * This is a setter for the launch parameters of the kernel, It is used in case
	 * those need to be changed for optimal performance or concurrency reasons
	 *
	 * @param gridConfig
	 */
	void setLaunchParams(LaunchParameters params);

	/**
	 *Retrieves the launch paramters for the kernel
	 * @return
	 */
	LaunchParameters getLaunchParams();

	friend std::ostream &operator<<(std::ostream &output, const AbstractElasticKernel &kernel);

};

#endif /* ABSTRACTELASTICKERNEL_HPP_ */
