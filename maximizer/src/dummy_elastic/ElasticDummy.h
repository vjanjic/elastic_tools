/**
 * ElasticDummy.h
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ELASTICDUMMY_H_
#define ELASTICDUMMY_H_

#include "../elastic_kernel/AbstractElasticKernel.hpp"
#include "../elastic_kernel/ConfigurationParameters.h"
#include "../elastic_kernel/occupancy_tools/OccupancyLimits.h"
#include "cudaCode/declarations.h"
#include <cuda.h>



class ElasticDummy: public AbstractElasticKernel {
public:
	ElasticDummy(gridParams_logical gridParL, blockParams_logical blockParL);
	virtual ~ElasticDummy();
	void initKernel();
	void runKernel();

};

#endif /* ELASTICDUMMY_H_ */
