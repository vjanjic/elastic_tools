/**
 * mainExample.cpp
 *
 * A file to illustrate how the software scheduler can be used
 *
 *
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "elastic_launcher/KernelScheduler.h"
#include "elastic_launcher/KernelExecutionQueue.h"
#include "stdio.h"
#include "cuda_runtime.h"
#include "misc/Macros.h"
#include "misc/SimpleTimer.h"
#include "occupancy_tools/OccupancyCalculator.h"
#include "misc/workloadGeneration.h"

/**
 * Runs an experiment with the supplied set of kernels and the specified optimization policy.
 * The experiment is run a specified amount of times and the average total execution time for
 * the kernels is returned
 *
 * @param policy the particular optimization policy
 * @param samples the number of times the experiment needs to be run
 * @return
 */
double RunExperimentWithPolicy(OptimizationPolicy policy, int samples) {
	double avg = 0;
	double N = (double) samples;
	for (int var = 0; var < samples; ++var) {
		KernelScheduler scheduler;
		addChunkingKernelsToScheduler(scheduler);
		avg = avg + scheduler.runKernels(policy);
	}
	return avg / N;
}

/**
 *
 * Prints GPU utilization statistics for a particular policy and the supplied set of
 * kernels
 *
 * @param policy the policy
 */
void printGPUUtilisationForPolicy(OptimizationPolicy policy) {

	KernelScheduler schl = KernelScheduler();

	addChunkingKernelsToScheduler(schl);

	GPUUtilization ut1 = schl.getGPUOccupancyForPolicy(policy);

	printf("Compute Occupancy: %.6f              |\n", ut1.averageComputeOccupancy);
	printf("Storage Occupancy: %.6f              |\n", ut1.averageStorageOccupancy);

}

/**
 * Prints GPU occupancy details for all the policies
 */
void printOptimisationPolicyDetails() {
	std::cout << "------------------NATIVE------------------" << std::endl;
	printGPUUtilisationForPolicy(NATIVE);
	std::cout << "------------------------------------------" << std::endl << std::endl;

	std::cout << "--------------------FAIR------------------" << std::endl;
	printGPUUtilisationForPolicy(FAIR);
	std::cout << "------------------------------------------" << std::endl << std::endl;

	std::cout << "---------FAIR_MAXIMUM_OCCUPANCY-----------" << std::endl;
	printGPUUtilisationForPolicy(FAIR_MAXIMUM_OCCUPANCY);
	std::cout << "------------------------------------------" << std::endl << std::endl;

	std::cout << "--------------MINIMUM_QUEUES--------------" << std::endl;
	printGPUUtilisationForPolicy(MINIMUM_QUEUES);
	std::cout << "------------------------------------------" << std::endl << std::endl;

	std::cout << "-----MINIMUM_QUEUES_MAXIMUM_OCCUPANCY-----" << std::endl;
	printGPUUtilisationForPolicy(MINIMUM_QUEUES_MAXIMUM_OCCUPANCY);
	std::cout << "------------------------------------------" << std::endl << std::endl;

	std::cout << "-------------MAXIMUM_CONCURENCY-----------" << std::endl;
	printGPUUtilisationForPolicy(MAXIMUM_CONCURENCY);
	std::cout << "------------------------------------------" << std::endl << std::endl;
}

/**
 *
 * Run experiments with all the policies and displays total execution time for each policy.
 * Number of samples can be specified
 *
 * @param samples number of samples
 */
void runAllPolicies(int samples) {
	std::cout << "native: " << RunExperimentWithPolicy(NATIVE, samples) << std::endl;
	//std::cout << "fair: " << RunExperimentWithPolicy(FAIR, samples) << std::endl;
	//std::cout << "fair_max_occ: " << RunExperimentWithPolicy(FAIR_MAXIMUM_OCCUPANCY, samples) << std::endl;
	//std::cout << "min_queues: " << RunExperimentWithPolicy(MINIMUM_QUEUES, samples) << std::endl;
	//std::cout << "min_queues_max_occ: " << RunExperimentWithPolicy(MINIMUM_QUEUES_MAXIMUM_OCCUPANCY, samples) << std::endl;
	//std::cout << "max_concurency: " << RunExperimentWithPolicy(MAXIMUM_CONCURENCY, samples) << std::endl;

}

/**
 * Prints the execution queues configuration for a particular policy
 *
 * @param policy the policy
 */
void printQueueConfigurationForPolicy(OptimizationPolicy policy) {
	KernelScheduler schl = KernelScheduler();
	addChunkingKernelsToScheduler(schl);
	schl.getGPUOccupancyForPolicy(policy);
	std::cout << schl << std::endl;
}

int main() {
	////printQueueConfigurationForPolicy(FAIR);
	printOptimisationPolicyDetails();
	//runAllPolicies(1);
}

