/**
 * starter.cpp
 *
 *  Created on: Jul 28, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */
#include "Chunking_elastic/elastic_wrapper/ElasticChunker.h"
#include "BlackScholes_elastic/elastic_wrapper/ElasticBSPricer.h"
#include  "ScalarProduct_elastic/elastic_wrapper/ElasticScalarProduct.h"
#include  "VectorAddition_elastic/elastic_wrapper/ElasticVectorAddition.h"
#include  "elastic_launcher/ElasticKernelMaker.h"
#include "elastic_launcher/KernelScheduler.h"
#include "elastic_launcher/KernelExecutionQueue.h"
#include "stdio.h"
#include "cuda_runtime.h"
#include "misc/Macros.h"
#include "misc/SimpleTimer.h"

struct config {
	int threads;
	int blocks;
};

void addKernelsToScheduler(KernelScheduler& schl) {
	schl.addKernel(makeElasticKernel(160, 1, VECTOR_ADD, "VECTOR_ADD__1", 14000000));
	schl.addKernel(makeElasticKernel(128, 16, MATRIX_MULT, "MATRIX_MULT__1", 2560));
	schl.addKernel(makeElasticKernel(160, 1, CHUNKING, "CHUNKING__1", 134217728));
	schl.addKernel(makeElasticKernel(128, 32, BLACK_SCHOLES, "BLACK_SCHOLES__1", 32500000));
	schl.addKernel(makeElasticKernel(64, 2, CHUNKING, "CHUNKING__2", 67108864));
	schl.addKernel(makeElasticKernel(512, 8, BLACK_SCHOLES, "BLACK_SCHOLES__2", 40000000));
	schl.addKernel(makeElasticKernel(512, 8, MATRIX_MULT, "MATRIX_MULT__2", 1024));
	schl.addKernel(makeElasticKernel(64, 8, CHUNKING, "CHUNKING__3", 67108864));
	schl.addKernel(makeElasticKernel(192, 16, CHUNKING, "CHUNKING__4", 134217728));
	schl.addKernel(makeElasticKernel(96, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__1", 22528));
	schl.addKernel(makeElasticKernel(64, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__2", 32768));
	schl.addKernel(makeElasticKernel(224, 8, VECTOR_ADD, "VECTOR_ADD__2", 10000000));
	schl.addKernel(makeElasticKernel(192, 2, SCALAR_PRODUCT, "SCALAR_PRODUCT__3", 12288));
	schl.addKernel(makeElasticKernel(128, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__4", 26624));
	schl.addKernel(makeElasticKernel(512, 16, VECTOR_ADD, "VECTOR_ADD__3", 5000000));
	schl.addKernel(makeElasticKernel(512, 32, VECTOR_ADD, "VECTOR_ADD__4", 15000000));
	schl.addKernel(makeElasticKernel(160, 8, SCALAR_PRODUCT, "SCALAR_PRODUCT__5", 6144));
	schl.addKernel(makeElasticKernel(256, 16, CHUNKING, "CHUNKING__5", 134217728));
	schl.addKernel(makeElasticKernel(64, 16, CHUNKING, "CHUNKING__6", 67108864));
	schl.addKernel(makeElasticKernel(96, 2, VECTOR_ADD, "VECTOR_ADD__5", 13000000));
	schl.addKernel(makeElasticKernel(192, 16, BLACK_SCHOLES, "BLACK_SCHOLES__3", 2500000));
	schl.addKernel(makeElasticKernel(64, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__6", 24576));
	schl.addKernel(makeElasticKernel(128, 2, MATRIX_MULT, "MATRIX_MULT__3", 2048));
	schl.addKernel(makeElasticKernel(192, 16, VECTOR_ADD, "VECTOR_ADD__6", 10000000));
	schl.addKernel(makeElasticKernel(224, 8, CHUNKING, "CHUNKING__7", 134217728));
	schl.addKernel(makeElasticKernel(32, 32, CHUNKING, "CHUNKING__8", 33554432));
	schl.addKernel(makeElasticKernel(64, 1, VECTOR_ADD, "VECTOR_ADD__7", 3000000));
	schl.addKernel(makeElasticKernel(512, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__7", 18432));
	schl.addKernel(makeElasticKernel(128, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__8", 8192));
	schl.addKernel(makeElasticKernel(96, 8, CHUNKING, "CHUNKING__9", 33554432));
	schl.addKernel(makeElasticKernel(512, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__9", 32768));
	schl.addKernel(makeElasticKernel(96, 4, VECTOR_ADD, "VECTOR_ADD__8", 4000000));
	schl.addKernel(makeElasticKernel(128, 32, BLACK_SCHOLES, "BLACK_SCHOLES__4", 12500000));
	schl.addKernel(makeElasticKernel(512, 2, MATRIX_MULT, "MATRIX_MULT__4", 2560));
	schl.addKernel(makeElasticKernel(32, 2, CHUNKING, "CHUNKING__10", 33554432));
	schl.addKernel(makeElasticKernel(192, 8, MATRIX_MULT, "MATRIX_MULT__5", 3584));
	schl.addKernel(makeElasticKernel(224, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__10", 24576));
	schl.addKernel(makeElasticKernel(512, 2, CHUNKING, "CHUNKING__11", 67108864));
	schl.addKernel(makeElasticKernel(256, 4, CHUNKING, "CHUNKING__12", 134217728));
	schl.addKernel(makeElasticKernel(224, 1, VECTOR_ADD, "VECTOR_ADD__9", 10000000));
	schl.addKernel(makeElasticKernel(128, 2, SCALAR_PRODUCT, "SCALAR_PRODUCT__11", 8192));
	schl.addKernel(makeElasticKernel(256, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__12", 10240));
	schl.addKernel(makeElasticKernel(192, 8, CHUNKING, "CHUNKING__13", 67108864));
	schl.addKernel(makeElasticKernel(32, 8, BLACK_SCHOLES, "BLACK_SCHOLES__5", 35000000));
	schl.addKernel(makeElasticKernel(96, 8, BLACK_SCHOLES, "BLACK_SCHOLES__6", 20000000));
	schl.addKernel(makeElasticKernel(256, 16, CHUNKING, "CHUNKING__14", 33554432));
	schl.addKernel(makeElasticKernel(32, 4, CHUNKING, "CHUNKING__15", 33554432));
	schl.addKernel(makeElasticKernel(192, 1, CHUNKING, "CHUNKING__16", 67108864));
	schl.addKernel(makeElasticKernel(64, 32, VECTOR_ADD, "VECTOR_ADD__10", 12000000));
	schl.addKernel(makeElasticKernel(512, 2, VECTOR_ADD, "VECTOR_ADD__11", 8000000));
	schl.addKernel(makeElasticKernel(32, 32, MATRIX_MULT, "MATRIX_MULT__6", 3584));
	schl.addKernel(makeElasticKernel(512, 1, VECTOR_ADD, "VECTOR_ADD__12", 18000000));
	schl.addKernel(makeElasticKernel(224, 2, VECTOR_ADD, "VECTOR_ADD__13", 7000000));
	schl.addKernel(makeElasticKernel(32, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__13", 18432));
	schl.addKernel(makeElasticKernel(192, 1, CHUNKING, "CHUNKING__17", 67108864));
	schl.addKernel(makeElasticKernel(64, 8, CHUNKING, "CHUNKING__18", 134217728));
	schl.addKernel(makeElasticKernel(192, 32, CHUNKING, "CHUNKING__19", 67108864));
	schl.addKernel(makeElasticKernel(224, 32, VECTOR_ADD, "VECTOR_ADD__14", 1000000));
	schl.addKernel(makeElasticKernel(96, 8, MATRIX_MULT, "MATRIX_MULT__7", 512));
	schl.addKernel(makeElasticKernel(256, 2, CHUNKING, "CHUNKING__20", 33554432));
	schl.addKernel(makeElasticKernel(256, 2, VECTOR_ADD, "VECTOR_ADD__15", 18000000));
	schl.addKernel(makeElasticKernel(32, 2, BLACK_SCHOLES, "BLACK_SCHOLES__7", 32500000));
	schl.addKernel(makeElasticKernel(512, 8, VECTOR_ADD, "VECTOR_ADD__16", 2000000));
	schl.addKernel(makeElasticKernel(64, 2, BLACK_SCHOLES, "BLACK_SCHOLES__8", 27500000));
	schl.addKernel(makeElasticKernel(224, 4, CHUNKING, "CHUNKING__21", 134217728));
	schl.addKernel(makeElasticKernel(224, 1, MATRIX_MULT, "MATRIX_MULT__8", 512));
	schl.addKernel(makeElasticKernel(512, 8, CHUNKING, "CHUNKING__22", 67108864));
	schl.addKernel(makeElasticKernel(128, 4, CHUNKING, "CHUNKING__23", 33554432));
	schl.addKernel(makeElasticKernel(64, 2, BLACK_SCHOLES, "BLACK_SCHOLES__9", 37500000));
	schl.addKernel(makeElasticKernel(96, 32, CHUNKING, "CHUNKING__24", 33554432));
	schl.addKernel(makeElasticKernel(64, 32, VECTOR_ADD, "VECTOR_ADD__17", 2000000));
	schl.addKernel(makeElasticKernel(64, 4, VECTOR_ADD, "VECTOR_ADD__18", 8000000));
	schl.addKernel(makeElasticKernel(160, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__14", 2048));
	schl.addKernel(makeElasticKernel(32, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__15", 32768));
	schl.addKernel(makeElasticKernel(128, 16, MATRIX_MULT, "MATRIX_MULT__9", 1536));
	schl.addKernel(makeElasticKernel(192, 8, VECTOR_ADD, "VECTOR_ADD__19", 6000000));
	schl.addKernel(makeElasticKernel(96, 4, MATRIX_MULT, "MATRIX_MULT__10", 3072));
	schl.addKernel(makeElasticKernel(224, 4, VECTOR_ADD, "VECTOR_ADD__20", 5000000));
	schl.addKernel(makeElasticKernel(32, 4, BLACK_SCHOLES, "BLACK_SCHOLES__10", 5000000));
	schl.addKernel(makeElasticKernel(192, 1, MATRIX_MULT, "MATRIX_MULT__11", 3584));
	schl.addKernel(makeElasticKernel(512, 2, BLACK_SCHOLES, "BLACK_SCHOLES__11", 30000000));
	schl.addKernel(makeElasticKernel(160, 8, VECTOR_ADD, "VECTOR_ADD__21", 17000000));
	schl.addKernel(makeElasticKernel(160, 16, MATRIX_MULT, "MATRIX_MULT__12", 3072));
	schl.addKernel(makeElasticKernel(512, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__16", 2048));
	schl.addKernel(makeElasticKernel(512, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__17", 26624));
	schl.addKernel(makeElasticKernel(32, 16, VECTOR_ADD, "VECTOR_ADD__22", 12000000));
	schl.addKernel(makeElasticKernel(160, 32, BLACK_SCHOLES, "BLACK_SCHOLES__12", 32500000));
	schl.addKernel(makeElasticKernel(256, 8, CHUNKING, "CHUNKING__25", 134217728));
	schl.addKernel(makeElasticKernel(64, 32, CHUNKING, "CHUNKING__26", 67108864));
	schl.addKernel(makeElasticKernel(192, 8, SCALAR_PRODUCT, "SCALAR_PRODUCT__18", 12288));
	schl.addKernel(makeElasticKernel(512, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__19", 14336));
	schl.addKernel(makeElasticKernel(64, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__20", 28672));
	schl.addKernel(makeElasticKernel(224, 16, VECTOR_ADD, "VECTOR_ADD__23", 9000000));
	schl.addKernel(makeElasticKernel(32, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__21", 16384));
	schl.addKernel(makeElasticKernel(96, 16, CHUNKING, "CHUNKING__27", 67108864));
	schl.addKernel(makeElasticKernel(160, 8, MATRIX_MULT, "MATRIX_MULT__13", 2048));
	schl.addKernel(makeElasticKernel(192, 2, BLACK_SCHOLES, "BLACK_SCHOLES__13", 2500000));
	schl.addKernel(makeElasticKernel(32, 4, VECTOR_ADD, "VECTOR_ADD__24", 18000000));
	schl.addKernel(makeElasticKernel(512, 16, CHUNKING, "CHUNKING__28", 134217728));
	schl.addKernel(makeElasticKernel(96, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__22", 6144));
	schl.addKernel(makeElasticKernel(160, 2, BLACK_SCHOLES, "BLACK_SCHOLES__14", 17500000));
	schl.addKernel(makeElasticKernel(224, 16, CHUNKING, "CHUNKING__29", 67108864));
	schl.addKernel(makeElasticKernel(32, 16, VECTOR_ADD, "VECTOR_ADD__25", 1000000));
	schl.addKernel(makeElasticKernel(128, 4, CHUNKING, "CHUNKING__30", 33554432));
	schl.addKernel(makeElasticKernel(96, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__23", 6144));
	schl.addKernel(makeElasticKernel(512, 8, VECTOR_ADD, "VECTOR_ADD__26", 7000000));
	schl.addKernel(makeElasticKernel(128, 32, BLACK_SCHOLES, "BLACK_SCHOLES__15", 17500000));
	schl.addKernel(makeElasticKernel(128, 32, BLACK_SCHOLES, "BLACK_SCHOLES__16", 22500000));
	schl.addKernel(makeElasticKernel(224, 8, VECTOR_ADD, "VECTOR_ADD__27", 13000000));
	schl.addKernel(makeElasticKernel(128, 4, MATRIX_MULT, "MATRIX_MULT__14", 2048));
	schl.addKernel(makeElasticKernel(128, 16, CHUNKING, "CHUNKING__31", 67108864));
	schl.addKernel(makeElasticKernel(128, 8, BLACK_SCHOLES, "BLACK_SCHOLES__17", 7500000));
	schl.addKernel(makeElasticKernel(224, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__24", 28672));
	schl.addKernel(makeElasticKernel(224, 32, VECTOR_ADD, "VECTOR_ADD__28", 18000000));
	schl.addKernel(makeElasticKernel(256, 32, MATRIX_MULT, "MATRIX_MULT__15", 1536));
	schl.addKernel(makeElasticKernel(512, 32, MATRIX_MULT, "MATRIX_MULT__16", 3584));
	schl.addKernel(makeElasticKernel(192, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__25", 30720));
	schl.addKernel(makeElasticKernel(192, 2, MATRIX_MULT, "MATRIX_MULT__17", 1536));
	schl.addKernel(makeElasticKernel(96, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__26", 18432));
	schl.addKernel(makeElasticKernel(96, 2, CHUNKING, "CHUNKING__32", 67108864));
	schl.addKernel(makeElasticKernel(128, 1, CHUNKING, "CHUNKING__33", 67108864));
	schl.addKernel(makeElasticKernel(96, 8, MATRIX_MULT, "MATRIX_MULT__18", 1536));
	schl.addKernel(makeElasticKernel(96, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__27", 20480));
	schl.addKernel(makeElasticKernel(128, 8, VECTOR_ADD, "VECTOR_ADD__29", 12000000));
	schl.addKernel(makeElasticKernel(160, 8, BLACK_SCHOLES, "BLACK_SCHOLES__18", 17500000));
	schl.addKernel(makeElasticKernel(224, 16, MATRIX_MULT, "MATRIX_MULT__19", 3072));
	schl.addKernel(makeElasticKernel(96, 32, BLACK_SCHOLES, "BLACK_SCHOLES__19", 40000000));
	schl.addKernel(makeElasticKernel(192, 4, MATRIX_MULT, "MATRIX_MULT__20", 1024));
	schl.addKernel(makeElasticKernel(160, 1, BLACK_SCHOLES, "BLACK_SCHOLES__20", 2500000));
	schl.addKernel(makeElasticKernel(160, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__28", 22528));
	schl.addKernel(makeElasticKernel(128, 2, BLACK_SCHOLES, "BLACK_SCHOLES__21", 27500000));
	schl.addKernel(makeElasticKernel(192, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__29", 20480));
	schl.addKernel(makeElasticKernel(256, 1, MATRIX_MULT, "MATRIX_MULT__21", 512));
	schl.addKernel(makeElasticKernel(192, 16, VECTOR_ADD, "VECTOR_ADD__30", 1000000));
	schl.addKernel(makeElasticKernel(160, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__30", 4096));
	schl.addKernel(makeElasticKernel(32, 16, MATRIX_MULT, "MATRIX_MULT__22", 3584));
	schl.addKernel(makeElasticKernel(256, 16, BLACK_SCHOLES, "BLACK_SCHOLES__22", 25000000));
	schl.addKernel(makeElasticKernel(224, 1, CHUNKING, "CHUNKING__34", 33554432));
	schl.addKernel(makeElasticKernel(32, 16, VECTOR_ADD, "VECTOR_ADD__31", 19000000));
	schl.addKernel(makeElasticKernel(160, 8, VECTOR_ADD, "VECTOR_ADD__32", 15000000));

}

double runFairExperiment(int itters) {
	double avg = 0;
	double N = (double) itters;
	for (int var = 0; var < itters; ++var) {
		KernelScheduler scheduler;
		addKernelsToScheduler(scheduler);
		SimpleTimer t("fair");
		t.start();
		scheduler.runKernels(FAIR);
		avg = avg + t.stop();

	}
	return avg / N;
}

double runMaxOccupancyExperiment(int itters) {
	double avg = 0;
	double N = (double) itters;
	for (int var = 0; var < itters; ++var) {

		KernelScheduler scheduler;
		addKernelsToScheduler(scheduler);
		SimpleTimer t("max_occupancy");
		t.start();
		scheduler.runKernels(MAXIMUM_OCCUPANCY);
		avg = avg + t.stop();
	}
	return avg / N;
}

double runMinQueuesExperiment(int itters) {
	double avg = 0;
	double N = (double) itters;
	for (int var = 0; var < itters; ++var) {
		KernelScheduler scheduler;
		addKernelsToScheduler(scheduler);
		SimpleTimer t("min_queues");
		t.start();
		scheduler.runKernels(MINIMUM_QUEUES);
		avg = avg + t.stop();
	}
	return avg / N;
}

int main() {

	std::cout << "fair: " << runFairExperiment(5) << std::endl;
	std::cout << "min_queues: " <<	runMinQueuesExperiment(5)<< std::endl;
	std::cout << "max_occupancy: " <<	runMaxOccupancyExperiment(5)<< std::endl;
}

