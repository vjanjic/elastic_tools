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

int main() {

	KernelScheduler scheduler;
	scheduler.addKernel(makeElasticKernel(352, 8, SCALAR_PRODUCT, "SCALAR_PRODUCT__1"));
	scheduler.addKernel(makeElasticKernel(320, 8, VECTOR_ADD, "VECTOR_ADD__1"));
	scheduler.addKernel(makeElasticKernel(32, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__2"));
	scheduler.addKernel(makeElasticKernel(128, 64, BLACK_SCHOLES, "BLACK_SCHOLES__1"));
	scheduler.addKernel(makeElasticKernel(160, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__3"));
	scheduler.addKernel(makeElasticKernel(96, 32, BLACK_SCHOLES, "BLACK_SCHOLES__2"));
	scheduler.addKernel(makeElasticKernel(256, 1, BLACK_SCHOLES, "BLACK_SCHOLES__3"));
	scheduler.addKernel(makeElasticKernel(32, 64, MATRIX_MULT, "MATRIX_MULT__1"));
	scheduler.addKernel(makeElasticKernel(320, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__4"));
	scheduler.addKernel(makeElasticKernel(64, 32, BLACK_SCHOLES, "BLACK_SCHOLES__4"));
	scheduler.addKernel(makeElasticKernel(288, 64, SCALAR_PRODUCT, "SCALAR_PRODUCT__5"));
	scheduler.addKernel(makeElasticKernel(160, 32, MATRIX_MULT, "MATRIX_MULT__2"));
	scheduler.addKernel(makeElasticKernel(448, 128, VECTOR_ADD, "VECTOR_ADD__2"));
	scheduler.addKernel(makeElasticKernel(32, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__6"));
	scheduler.addKernel(makeElasticKernel(256, 1, BLACK_SCHOLES, "BLACK_SCHOLES__5"));
	scheduler.addKernel(makeElasticKernel(32, 16, BLACK_SCHOLES, "BLACK_SCHOLES__6"));
	scheduler.addKernel(makeElasticKernel(64, 1, VECTOR_ADD, "VECTOR_ADD__3"));
	scheduler.addKernel(makeElasticKernel(192, 64, CHUNKING, "CHUNKING__1"));
	scheduler.addKernel(makeElasticKernel(384, 64, SCALAR_PRODUCT, "SCALAR_PRODUCT__7"));
	scheduler.addKernel(makeElasticKernel(320, 128, VECTOR_ADD, "VECTOR_ADD__4"));
	scheduler.addKernel(makeElasticKernel(352, 4, CHUNKING, "CHUNKING__2"));
	scheduler.addKernel(makeElasticKernel(320, 16, MATRIX_MULT, "MATRIX_MULT__3"));
	scheduler.addKernel(makeElasticKernel(224, 2, CHUNKING, "CHUNKING__3"));
	scheduler.addKernel(makeElasticKernel(480, 8, BLACK_SCHOLES, "BLACK_SCHOLES__7"));
	scheduler.addKernel(makeElasticKernel(64, 2, MATRIX_MULT, "MATRIX_MULT__4"));
	scheduler.addKernel(makeElasticKernel(224, 128, MATRIX_MULT, "MATRIX_MULT__5"));
	scheduler.addKernel(makeElasticKernel(64, 4, BLACK_SCHOLES, "BLACK_SCHOLES__8"));
	scheduler.addKernel(makeElasticKernel(224, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__8"));
	scheduler.addKernel(makeElasticKernel(480, 8, VECTOR_ADD, "VECTOR_ADD__5"));
	scheduler.addKernel(makeElasticKernel(64, 1, CHUNKING, "CHUNKING__4"));
	scheduler.addKernel(makeElasticKernel(416, 2, SCALAR_PRODUCT, "SCALAR_PRODUCT__9"));
	scheduler.addKernel(makeElasticKernel(352, 8, VECTOR_ADD, "VECTOR_ADD__6"));
	scheduler.addKernel(makeElasticKernel(160, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__10"));
	scheduler.addKernel(makeElasticKernel(224, 1, CHUNKING, "CHUNKING__5"));
	scheduler.addKernel(makeElasticKernel(160, 32, BLACK_SCHOLES, "BLACK_SCHOLES__9"));
	scheduler.addKernel(makeElasticKernel(352, 1, BLACK_SCHOLES, "BLACK_SCHOLES__10"));
	scheduler.addKernel(makeElasticKernel(32, 1, VECTOR_ADD, "VECTOR_ADD__7"));
	scheduler.addKernel(makeElasticKernel(96, 64, MATRIX_MULT, "MATRIX_MULT__6"));
	scheduler.addKernel(makeElasticKernel(480, 64, SCALAR_PRODUCT, "SCALAR_PRODUCT__11"));
	scheduler.addKernel(makeElasticKernel(384, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__12"));
	scheduler.addKernel(makeElasticKernel(512, 1, MATRIX_MULT, "MATRIX_MULT__7"));
	scheduler.addKernel(makeElasticKernel(288, 128, MATRIX_MULT, "MATRIX_MULT__8"));
	scheduler.addKernel(makeElasticKernel(32, 1, CHUNKING, "CHUNKING__6"));
	scheduler.addKernel(makeElasticKernel(128, 8, VECTOR_ADD, "VECTOR_ADD__8"));
	scheduler.addKernel(makeElasticKernel(448, 64, CHUNKING, "CHUNKING__7"));
	scheduler.addKernel(makeElasticKernel(384, 128, VECTOR_ADD, "VECTOR_ADD__9"));
	scheduler.addKernel(makeElasticKernel(352, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__13"));
	scheduler.addKernel(makeElasticKernel(160, 128, SCALAR_PRODUCT, "SCALAR_PRODUCT__14"));
	scheduler.addKernel(makeElasticKernel(480, 2, BLACK_SCHOLES, "BLACK_SCHOLES__11"));
	scheduler.addKernel(makeElasticKernel(128, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__15"));

	scheduler.runKernels();

}

