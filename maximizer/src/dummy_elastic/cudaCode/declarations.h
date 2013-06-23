/**
 * declarations.h
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef DECLARATIONS_H_
#define DECLARATIONS_H_


#include <cuda_runtime.h>
#include "../../elastic_kernel/ConfigurationParameters.h"
#include "../../elastic_kernel/occupancy_tools/OccupancyData.h"
#include "../../elastic_kernel/occupancy_tools/OccupancyCalculator.h"

extern "C" PhysicalConfiguration scale_dummy_elastic(blockParams_logical blk_logical, gridParams_logical grd_logical, KernelLimits limits);
extern "C" void lauch_dummy_elastic(blockParams_logical blk_logical, gridParams_logical grd_logical, KernelLimits limits);


#endif /* DECLARATIONS_H_ */
