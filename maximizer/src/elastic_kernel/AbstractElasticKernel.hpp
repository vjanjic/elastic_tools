/**
 * AbstractElasticKernel.hpp
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef ABSTRACTELASTICKERNEL_HPP_
#define ABSTRACTELASTICKERNEL_HPP_

#include "ConfigurationParameters.h"
#include "stdint.h"

class AbstractElasticKernel {

protected:
	gridParams_logical lGrid;
	blockParams_logical lBlock;

public:
	AbstractElasticKernel(gridParams_logical gridParL, blockParams_logical blockParL);
	AbstractElasticKernel(size_t gdx, size_t gdy, size_t blx, size_t bly, size_t blz);
	virtual ~AbstractElasticKernel();

	virtual void initKernel() = 0;
	virtual void runKernel() = 0;

};

#endif /* ABSTRACTELASTICKERNEL_HPP_ */
