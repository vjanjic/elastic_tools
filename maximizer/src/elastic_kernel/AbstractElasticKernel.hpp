/**
 * AbstractElasticKernel.hpp
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
	LaunchParameters gridConfig;
	std::string name;

public:
	AbstractElasticKernel();

	AbstractElasticKernel(const LaunchParameters& gridConfig,std::string name);

	virtual ~AbstractElasticKernel();
	virtual void initKernel() = 0;
	virtual void runKernel(cudaStream_t &streamToRunIn) = 0;
	virtual cudaFuncAttributes getKernelProperties() = 0;
	virtual void freeResources() = 0;

	void setLaunchlParams(const LaunchParameters& gridConfig);

	LaunchParameters getLaunchParams();

	friend std::ostream &operator<<(std::ostream &output, const AbstractElasticKernel &kernel);


};

#endif /* ABSTRACTELASTICKERNEL_HPP_ */
