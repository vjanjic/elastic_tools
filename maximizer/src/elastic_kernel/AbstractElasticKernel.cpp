/**
 * AbstractElasticKernel.cpp
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "AbstractElasticKernel.hpp"

AbstractElasticKernel::AbstractElasticKernel() {
	LaunchParameters newParams = LaunchParameters();
	this->gridConfig = newParams;
	this->name = "n/a";
}

AbstractElasticKernel::AbstractElasticKernel(const LaunchParameters& gridConfig,std::string name) {
	this->gridConfig = gridConfig;
	this->name = name;
}

AbstractElasticKernel::~AbstractElasticKernel() {
}

void AbstractElasticKernel::setLaunchlParams(const LaunchParameters& gridConfig) {
	this->gridConfig = gridConfig;

}

LaunchParameters AbstractElasticKernel::getLaunchParams() {
	return this->gridConfig;
}

std::ostream & operator<<(std::ostream &output, const AbstractElasticKernel &kernel) {
	output << "[" << kernel.name << "] @ " << kernel.gridConfig;
	return output;
}
