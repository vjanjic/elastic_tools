/**
 * SimpleTimer.cpp
 *
 *  Created on: Aug 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "SimpleTimer.h"

SimpleTimer::SimpleTimer(std::string name) {
	this->name = name;

}

void SimpleTimer::start() {\
	this->start_ = clock();
}

double SimpleTimer::stop() {
	this->end_ = clock();
	double elapsedTime = double(this->end_ - this->start_) / CLOCKS_PER_SEC;
	//printf ("[%s] [%.2lf s]\n", elapsedTime,this->name.c_str() );
	return elapsedTime;

}

SimpleTimer::~SimpleTimer() {
	// TODO Auto-generated destructor stub
}

