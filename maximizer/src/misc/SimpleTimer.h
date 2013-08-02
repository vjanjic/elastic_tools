/**
 * SimpleTimer.h
 *
 *  Created on: Aug 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef SIMPLETIMER_H_
#define SIMPLETIMER_H_
#include "stdio.h"
#include  <ctime>
#include <string>

class SimpleTimer {
private:
	clock_t start_;
	clock_t end_;
	std::string name;

public:
	SimpleTimer(std::string name);
	void start();
	double stop();

	virtual ~SimpleTimer();
};

#endif /* SIMPLETIMER_H_ */
