/**
 * AbstractElasticKernel.cpp
 *
 *  Created on: Jun 22, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "AbstractElasticKernel.hpp"

//really nothing complicated here..

AbstractElasticKernel::AbstractElasticKernel() {
	LaunchParameters newParams = LaunchParameters();
	this->gridConfig = newParams;
	this->name = "n/a";
}

AbstractElasticKernel::AbstractElasticKernel(const LaunchParameters& gridConfig, std::string name) {
	this->gridConfig = gridConfig;
	this->name = name;
}

AbstractElasticKernel::~AbstractElasticKernel() {
}

LaunchParameters AbstractElasticKernel::getLaunchParams() {
	return this->gridConfig;
}

void AbstractElasticKernel::setLaunchParams(LaunchParameters params) {
	this->gridConfig = params;
}

std::ostream & operator<<(std::ostream &output, const AbstractElasticKernel &kernel) {
	output << "[" << kernel.name << "] @ " << kernel.gridConfig << " [" << kernel.memConsumption << "]";
	return output;
}
