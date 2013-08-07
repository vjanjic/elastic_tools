/**
 * SimpleTimer.h
 *
 * Just a simple timer that is used to time the experiments
 *
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
	double elapsedTime;
	std::string name;

public:
	/**
	 * Initializes the private variables and assigns a name to the timer
	 *
	 * @param name
	 */
	SimpleTimer(std::string name) {
		this->name = name;
		this->start_ = 0;
		this->end_ = 0;
		this->elapsedTime = 0;
	}

	virtual ~SimpleTimer() {
		//dont need qnything here
	}

	/**
	 * Starts the timer
	 */
	void start() {
		this->start_ = clock();
	}
	/**
	 * Stop the timer and calculate elapsed time since start
	 *
	 * @return the elapsed time in seconds
	 */
	double stop() {
		this->end_ = clock();
		this->elapsedTime = double(this->end_ - this->start_) / CLOCKS_PER_SEC;
		//printf ("[%s] [%.2lf s]\n", elapsedTime,this->name.c_str() );
		return this->elapsedTime;
	}

	/**
	 * Returns the elapsed time
	 *
	 * @return elapsed time
	 */
	double getElapsedTime() {
		return this->elapsedTime;
	}

};

#endif /* SIMPLETIMER_H_ */
