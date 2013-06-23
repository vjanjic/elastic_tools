/**
 * AbstractElasticKernel.cpp
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "AbstractElasticKernel.hpp"

AbstractElasticKernel::AbstractElasticKernel(gridParams_logical gridParL, blockParams_logical blockParL) {

	this->lBlock = blockParL;
	this->lGrid = gridParL;

}

AbstractElasticKernel::AbstractElasticKernel(size_t gdx, size_t gdy, size_t blx, size_t bly, size_t blz) {
	this->lBlock = getLogicalBlock(blx, bly, blz);
	this->lGrid = getLogicalGrid(gdx, gdy);
}

AbstractElasticKernel::~AbstractElasticKernel() {
	// TODO Auto-generated destructor stub
}

