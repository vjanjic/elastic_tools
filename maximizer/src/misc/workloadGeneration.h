/**
 * workloadGeneration.h
 *
 * Just a function that adds pre-generated workload to a kernel scheduler
 *
 *  Created on: Aug 8, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef WORKLOADGENERATION_H_
#define WORKLOADGENERATION_H_
#include  "../elastic_launcher/ElasticKernelMaker.h"

/**
 *
 * Given a reference to a kernel scheduler, this function adds
 * some pre-generated workload in the form of different kernels
 *
 * @param schl a reference to the kernel scheduler to be filler with workload
 */
/*void addKernelsToScheduler(KernelScheduler& schl) {
	schl.addKernel(makeElasticKernel(160, 2, VECTOR_ADD, "VECTOR_ADD__1", 14000000));
	schl.addKernel(makeElasticKernel(128, 128, MATRIX_MULT, "MATRIX_MULT__1", 2560));
	schl.addKernel(makeElasticKernel(160, 1, CHUNKING, "CHUNKING__1", 134217728));
	schl.addKernel(makeElasticKernel(128, 512, BLACK_SCHOLES, "BLACK_SCHOLES__1", 32500000));
	schl.addKernel(makeElasticKernel(64, 4, CHUNKING, "CHUNKING__2", 67108864));
	schl.addKernel(makeElasticKernel(512, 64, BLACK_SCHOLES, "BLACK_SCHOLES__2", 40000000));
	schl.addKernel(makeElasticKernel(512, 32, MATRIX_MULT, "MATRIX_MULT__2", 1024));
	schl.addKernel(makeElasticKernel(64, 64, CHUNKING, "CHUNKING__3", 67108864));
	schl.addKernel(makeElasticKernel(192, 128, CHUNKING, "CHUNKING__4", 134217728));
	schl.addKernel(makeElasticKernel(96, 512, SCALAR_PRODUCT, "SCALAR_PRODUCT__1", 22528));
	schl.addKernel(makeElasticKernel(64, 512, SCALAR_PRODUCT, "SCALAR_PRODUCT__2", 32768));
	schl.addKernel(makeElasticKernel(224, 64, VECTOR_ADD, "VECTOR_ADD__2", 10000000));
	schl.addKernel(makeElasticKernel(192, 4, SCALAR_PRODUCT, "SCALAR_PRODUCT__3", 12288));
	schl.addKernel(makeElasticKernel(128, 256, SCALAR_PRODUCT, "SCALAR_PRODUCT__4", 26624));
	schl.addKernel(makeElasticKernel(512, 256, VECTOR_ADD, "VECTOR_ADD__3", 5000000));
	schl.addKernel(makeElasticKernel(512, 256, VECTOR_ADD, "VECTOR_ADD__4", 15000000));
	schl.addKernel(makeElasticKernel(160, 32, SCALAR_PRODUCT, "SCALAR_PRODUCT__5", 6144));
	schl.addKernel(makeElasticKernel(256, 64, CHUNKING, "CHUNKING__5", 134217728));
	schl.addKernel(makeElasticKernel(64, 128, CHUNKING, "CHUNKING__6", 67108864));
	schl.addKernel(makeElasticKernel(96, 4, VECTOR_ADD, "VECTOR_ADD__5", 13000000));
	schl.addKernel(makeElasticKernel(192, 128, BLACK_SCHOLES, "BLACK_SCHOLES__3", 2500000));
	schl.addKernel(makeElasticKernel(64, 2, SCALAR_PRODUCT, "SCALAR_PRODUCT__6", 24576));
	schl.addKernel(makeElasticKernel(128, 4, MATRIX_MULT, "MATRIX_MULT__3", 2048));
	schl.addKernel(makeElasticKernel(192, 128, VECTOR_ADD, "VECTOR_ADD__6", 10000000));
	schl.addKernel(makeElasticKernel(224, 32, CHUNKING, "CHUNKING__7", 134217728));
	schl.addKernel(makeElasticKernel(32, 256, CHUNKING, "CHUNKING__8", 33554432));
	schl.addKernel(makeElasticKernel(64, 1, VECTOR_ADD, "VECTOR_ADD__7", 3000000));
	schl.addKernel(makeElasticKernel(512, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__7", 18432));
	schl.addKernel(makeElasticKernel(128, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__8", 8192));
	schl.addKernel(makeElasticKernel(96, 32, CHUNKING, "CHUNKING__9", 33554432));
	schl.addKernel(makeElasticKernel(512, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__9", 32768));
	schl.addKernel(makeElasticKernel(96, 16, VECTOR_ADD, "VECTOR_ADD__8", 4000000));
	schl.addKernel(makeElasticKernel(128, 512, BLACK_SCHOLES, "BLACK_SCHOLES__4", 12500000));
	schl.addKernel(makeElasticKernel(512, 8, MATRIX_MULT, "MATRIX_MULT__4", 2560));
	schl.addKernel(makeElasticKernel(32, 4, CHUNKING, "CHUNKING__10", 33554432));
	schl.addKernel(makeElasticKernel(192, 64, MATRIX_MULT, "MATRIX_MULT__5", 3584));
	schl.addKernel(makeElasticKernel(224, 8, SCALAR_PRODUCT, "SCALAR_PRODUCT__10", 24576));
	schl.addKernel(makeElasticKernel(512, 2, CHUNKING, "CHUNKING__11", 67108864));
	schl.addKernel(makeElasticKernel(256, 8, CHUNKING, "CHUNKING__12", 134217728));
	schl.addKernel(makeElasticKernel(224, 1, VECTOR_ADD, "VECTOR_ADD__9", 10000000));
	schl.addKernel(makeElasticKernel(128, 2, SCALAR_PRODUCT, "SCALAR_PRODUCT__11", 8192));
	schl.addKernel(makeElasticKernel(256, 2, SCALAR_PRODUCT, "SCALAR_PRODUCT__12", 10240));
	schl.addKernel(makeElasticKernel(192, 64, CHUNKING, "CHUNKING__13", 67108864));
	schl.addKernel(makeElasticKernel(32, 32, BLACK_SCHOLES, "BLACK_SCHOLES__5", 35000000));
	schl.addKernel(makeElasticKernel(96, 32, BLACK_SCHOLES, "BLACK_SCHOLES__6", 20000000));
	schl.addKernel(makeElasticKernel(256, 128, CHUNKING, "CHUNKING__14", 33554432));
	schl.addKernel(makeElasticKernel(32, 16, CHUNKING, "CHUNKING__15", 33554432));
	schl.addKernel(makeElasticKernel(192, 1, CHUNKING, "CHUNKING__16", 67108864));
	schl.addKernel(makeElasticKernel(64, 512, VECTOR_ADD, "VECTOR_ADD__10", 12000000));
	schl.addKernel(makeElasticKernel(512, 2, VECTOR_ADD, "VECTOR_ADD__11", 8000000));
	schl.addKernel(makeElasticKernel(32, 256, MATRIX_MULT, "MATRIX_MULT__6", 3584));
	schl.addKernel(makeElasticKernel(512, 2, VECTOR_ADD, "VECTOR_ADD__12", 18000000));
	schl.addKernel(makeElasticKernel(224, 4, VECTOR_ADD, "VECTOR_ADD__13", 7000000));
	schl.addKernel(makeElasticKernel(32, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__13", 18432));
	schl.addKernel(makeElasticKernel(192, 2, CHUNKING, "CHUNKING__17", 67108864));
	schl.addKernel(makeElasticKernel(64, 32, CHUNKING, "CHUNKING__18", 134217728));
	schl.addKernel(makeElasticKernel(192, 256, CHUNKING, "CHUNKING__19", 67108864));
	schl.addKernel(makeElasticKernel(224, 512, VECTOR_ADD, "VECTOR_ADD__14", 1000000));
	schl.addKernel(makeElasticKernel(96, 32, MATRIX_MULT, "MATRIX_MULT__7", 512));
	schl.addKernel(makeElasticKernel(256, 4, CHUNKING, "CHUNKING__20", 33554432));
	schl.addKernel(makeElasticKernel(256, 8, VECTOR_ADD, "VECTOR_ADD__15", 18000000));
	schl.addKernel(makeElasticKernel(32, 2, BLACK_SCHOLES, "BLACK_SCHOLES__7", 32500000));
	schl.addKernel(makeElasticKernel(512, 64, VECTOR_ADD, "VECTOR_ADD__16", 2000000));
	schl.addKernel(makeElasticKernel(64, 8, BLACK_SCHOLES, "BLACK_SCHOLES__8", 27500000));
	schl.addKernel(makeElasticKernel(224, 16, CHUNKING, "CHUNKING__21", 134217728));
	schl.addKernel(makeElasticKernel(224, 2, MATRIX_MULT, "MATRIX_MULT__8", 512));
	schl.addKernel(makeElasticKernel(512, 32, CHUNKING, "CHUNKING__22", 67108864));
	schl.addKernel(makeElasticKernel(128, 16, CHUNKING, "CHUNKING__23", 33554432));
	schl.addKernel(makeElasticKernel(64, 4, BLACK_SCHOLES, "BLACK_SCHOLES__9", 37500000));
	schl.addKernel(makeElasticKernel(96, 512, CHUNKING, "CHUNKING__24", 33554432));
	schl.addKernel(makeElasticKernel(64, 256, VECTOR_ADD, "VECTOR_ADD__17", 2000000));
	schl.addKernel(makeElasticKernel(64, 8, VECTOR_ADD, "VECTOR_ADD__18", 8000000));
	schl.addKernel(makeElasticKernel(160, 128, SCALAR_PRODUCT, "SCALAR_PRODUCT__14", 2048));
	schl.addKernel(makeElasticKernel(32, 256, SCALAR_PRODUCT, "SCALAR_PRODUCT__15", 32768));
	schl.addKernel(makeElasticKernel(128, 128, MATRIX_MULT, "MATRIX_MULT__9", 1536));
	schl.addKernel(makeElasticKernel(192, 32, VECTOR_ADD, "VECTOR_ADD__19", 6000000));
	schl.addKernel(makeElasticKernel(96, 8, MATRIX_MULT, "MATRIX_MULT__10", 3072));
	schl.addKernel(makeElasticKernel(224, 8, VECTOR_ADD, "VECTOR_ADD__20", 5000000));
	schl.addKernel(makeElasticKernel(32, 16, BLACK_SCHOLES, "BLACK_SCHOLES__10", 5000000));
	schl.addKernel(makeElasticKernel(192, 1, MATRIX_MULT, "MATRIX_MULT__11", 3584));
	schl.addKernel(makeElasticKernel(512, 4, BLACK_SCHOLES, "BLACK_SCHOLES__11", 30000000));
	schl.addKernel(makeElasticKernel(160, 32, VECTOR_ADD, "VECTOR_ADD__21", 17000000));
	schl.addKernel(makeElasticKernel(160, 64, MATRIX_MULT, "MATRIX_MULT__12", 3072));
	schl.addKernel(makeElasticKernel(512, 8, SCALAR_PRODUCT, "SCALAR_PRODUCT__16", 2048));
	schl.addKernel(makeElasticKernel(512, 512, SCALAR_PRODUCT, "SCALAR_PRODUCT__17", 26624));
	schl.addKernel(makeElasticKernel(32, 64, VECTOR_ADD, "VECTOR_ADD__22", 12000000));
	schl.addKernel(makeElasticKernel(160, 512, BLACK_SCHOLES, "BLACK_SCHOLES__12", 32500000));
	schl.addKernel(makeElasticKernel(256, 32, CHUNKING, "CHUNKING__25", 134217728));
	schl.addKernel(makeElasticKernel(64, 512, CHUNKING, "CHUNKING__26", 67108864));
	schl.addKernel(makeElasticKernel(192, 64, SCALAR_PRODUCT, "SCALAR_PRODUCT__18", 12288));
	schl.addKernel(makeElasticKernel(512, 128, SCALAR_PRODUCT, "SCALAR_PRODUCT__19", 14336));
	schl.addKernel(makeElasticKernel(64, 512, SCALAR_PRODUCT, "SCALAR_PRODUCT__20", 28672));
	schl.addKernel(makeElasticKernel(224, 128, VECTOR_ADD, "VECTOR_ADD__23", 9000000));
	schl.addKernel(makeElasticKernel(32, 512, SCALAR_PRODUCT, "SCALAR_PRODUCT__21", 16384));
	schl.addKernel(makeElasticKernel(96, 128, CHUNKING, "CHUNKING__27", 67108864));
	schl.addKernel(makeElasticKernel(160, 64, MATRIX_MULT, "MATRIX_MULT__13", 2048));
	schl.addKernel(makeElasticKernel(192, 4, BLACK_SCHOLES, "BLACK_SCHOLES__13", 2500000));
	schl.addKernel(makeElasticKernel(32, 16, VECTOR_ADD, "VECTOR_ADD__24", 18000000));
	schl.addKernel(makeElasticKernel(512, 128, CHUNKING, "CHUNKING__28", 134217728));
	schl.addKernel(makeElasticKernel(96, 128, SCALAR_PRODUCT, "SCALAR_PRODUCT__22", 6144));
	schl.addKernel(makeElasticKernel(160, 8, BLACK_SCHOLES, "BLACK_SCHOLES__14", 17500000));
	schl.addKernel(makeElasticKernel(224, 128, CHUNKING, "CHUNKING__29", 67108864));
	schl.addKernel(makeElasticKernel(32, 128, VECTOR_ADD, "VECTOR_ADD__25", 1000000));
	schl.addKernel(makeElasticKernel(128, 16, CHUNKING, "CHUNKING__30", 33554432));
	schl.addKernel(makeElasticKernel(96, 64, SCALAR_PRODUCT, "SCALAR_PRODUCT__23", 6144));
	schl.addKernel(makeElasticKernel(512, 64, VECTOR_ADD, "VECTOR_ADD__26", 7000000));
	schl.addKernel(makeElasticKernel(128, 512, BLACK_SCHOLES, "BLACK_SCHOLES__15", 17500000));
	schl.addKernel(makeElasticKernel(128, 512, BLACK_SCHOLES, "BLACK_SCHOLES__16", 22500000));
	schl.addKernel(makeElasticKernel(224, 64, VECTOR_ADD, "VECTOR_ADD__27", 13000000));
	schl.addKernel(makeElasticKernel(128, 16, MATRIX_MULT, "MATRIX_MULT__14", 2048));
	schl.addKernel(makeElasticKernel(128, 128, CHUNKING, "CHUNKING__31", 67108864));
	schl.addKernel(makeElasticKernel(128, 32, BLACK_SCHOLES, "BLACK_SCHOLES__17", 7500000));
	schl.addKernel(makeElasticKernel(224, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__24", 28672));
	schl.addKernel(makeElasticKernel(224, 256, VECTOR_ADD, "VECTOR_ADD__28", 18000000));
	schl.addKernel(makeElasticKernel(256, 512, MATRIX_MULT, "MATRIX_MULT__15", 1536));
	schl.addKernel(makeElasticKernel(512, 512, MATRIX_MULT, "MATRIX_MULT__16", 3584));
	schl.addKernel(makeElasticKernel(192, 128, SCALAR_PRODUCT, "SCALAR_PRODUCT__25", 30720));
	schl.addKernel(makeElasticKernel(192, 4, MATRIX_MULT, "MATRIX_MULT__17", 1536));
	schl.addKernel(makeElasticKernel(96, 2, SCALAR_PRODUCT, "SCALAR_PRODUCT__26", 18432));
	schl.addKernel(makeElasticKernel(96, 4, CHUNKING, "CHUNKING__32", 67108864));
	schl.addKernel(makeElasticKernel(128, 2, CHUNKING, "CHUNKING__33", 67108864));
	schl.addKernel(makeElasticKernel(96, 64, MATRIX_MULT, "MATRIX_MULT__18", 1536));
	schl.addKernel(makeElasticKernel(96, 512, SCALAR_PRODUCT, "SCALAR_PRODUCT__27", 20480));
	schl.addKernel(makeElasticKernel(128, 32, VECTOR_ADD, "VECTOR_ADD__29", 12000000));
	schl.addKernel(makeElasticKernel(160, 32, BLACK_SCHOLES, "BLACK_SCHOLES__18", 17500000));
	schl.addKernel(makeElasticKernel(224, 128, MATRIX_MULT, "MATRIX_MULT__19", 3072));
	schl.addKernel(makeElasticKernel(96, 512, BLACK_SCHOLES, "BLACK_SCHOLES__19", 40000000));
	schl.addKernel(makeElasticKernel(192, 16, MATRIX_MULT, "MATRIX_MULT__20", 1024));
	schl.addKernel(makeElasticKernel(160, 2, BLACK_SCHOLES, "BLACK_SCHOLES__20", 2500000));
	schl.addKernel(makeElasticKernel(160, 128, SCALAR_PRODUCT, "SCALAR_PRODUCT__28", 22528));
	schl.addKernel(makeElasticKernel(128, 8, BLACK_SCHOLES, "BLACK_SCHOLES__21", 27500000));
	schl.addKernel(makeElasticKernel(192, 16, SCALAR_PRODUCT, "SCALAR_PRODUCT__29", 20480));
	schl.addKernel(makeElasticKernel(256, 1, MATRIX_MULT, "MATRIX_MULT__21", 512));
	schl.addKernel(makeElasticKernel(192, 128, VECTOR_ADD, "VECTOR_ADD__30", 1000000));
	schl.addKernel(makeElasticKernel(160, 1, SCALAR_PRODUCT, "SCALAR_PRODUCT__30", 4096));
	schl.addKernel(makeElasticKernel(32, 128, MATRIX_MULT, "MATRIX_MULT__22", 3584));
	schl.addKernel(makeElasticKernel(256, 128, BLACK_SCHOLES, "BLACK_SCHOLES__22", 25000000));
	schl.addKernel(makeElasticKernel(224, 1, CHUNKING, "CHUNKING__34", 33554432));
	schl.addKernel(makeElasticKernel(32, 128, VECTOR_ADD, "VECTOR_ADD__31", 19000000));
	schl.addKernel(makeElasticKernel(160, 32, VECTOR_ADD, "VECTOR_ADD__32", 15000000));

}*/

/**
 *
 *
 *
 * @param schl
 */
void addChunkingKernelsToScheduler(KernelScheduler& schl) {

	schl.addKernel(makeElasticKernel(160,2,CHUNKING,"CHUNKING__1", 134217728));
	schl.addKernel(makeElasticKernel(224,128,CHUNKING,"CHUNKING__2", 67108864));
	schl.addKernel(makeElasticKernel(160,256,CHUNKING,"CHUNKING__3", 33554432));
	schl.addKernel(makeElasticKernel(32,16,CHUNKING,"CHUNKING__4", 134217728));
	schl.addKernel(makeElasticKernel(128,512,CHUNKING,"CHUNKING__5", 33554432));
	schl.addKernel(makeElasticKernel(64,128,CHUNKING,"CHUNKING__6", 33554432));
	schl.addKernel(makeElasticKernel(128,2,CHUNKING,"CHUNKING__7", 67108864));
	schl.addKernel(makeElasticKernel(64,512,CHUNKING,"CHUNKING__8", 134217728));
	schl.addKernel(makeElasticKernel(512,32,CHUNKING,"CHUNKING__9", 134217728));
	schl.addKernel(makeElasticKernel(192,2,CHUNKING,"CHUNKING__10", 33554432));
	schl.addKernel(makeElasticKernel(128,1,CHUNKING,"CHUNKING__11", 134217728));
	schl.addKernel(makeElasticKernel(64,32,CHUNKING,"CHUNKING__12", 134217728));
	schl.addKernel(makeElasticKernel(96,512,CHUNKING,"CHUNKING__13", 67108864));
	schl.addKernel(makeElasticKernel(512,64,CHUNKING,"CHUNKING__14", 33554432));
	schl.addKernel(makeElasticKernel(512,32,CHUNKING,"CHUNKING__15", 67108864));
	schl.addKernel(makeElasticKernel(224,128,CHUNKING,"CHUNKING__16", 67108864));
	schl.addKernel(makeElasticKernel(192,4,CHUNKING,"CHUNKING__17", 67108864));
	schl.addKernel(makeElasticKernel(256,8,CHUNKING,"CHUNKING__18", 67108864));
	schl.addKernel(makeElasticKernel(224,16,CHUNKING,"CHUNKING__19", 134217728));
	schl.addKernel(makeElasticKernel(192,256,CHUNKING,"CHUNKING__20", 33554432));
	schl.addKernel(makeElasticKernel(512,256,CHUNKING,"CHUNKING__21", 134217728));
	schl.addKernel(makeElasticKernel(160,128,CHUNKING,"CHUNKING__22", 67108864));
	schl.addKernel(makeElasticKernel(64,32,CHUNKING,"CHUNKING__23", 134217728));
	schl.addKernel(makeElasticKernel(64,256,CHUNKING,"CHUNKING__24", 134217728));
	schl.addKernel(makeElasticKernel(64,128,CHUNKING,"CHUNKING__25", 33554432));
	schl.addKernel(makeElasticKernel(64,32,CHUNKING,"CHUNKING__26", 33554432));
	schl.addKernel(makeElasticKernel(192,64,CHUNKING,"CHUNKING__27", 134217728));
	schl.addKernel(makeElasticKernel(128,32,CHUNKING,"CHUNKING__28", 33554432));
	schl.addKernel(makeElasticKernel(64,2,CHUNKING,"CHUNKING__29", 67108864));
	schl.addKernel(makeElasticKernel(64,128,CHUNKING,"CHUNKING__30", 67108864));
	schl.addKernel(makeElasticKernel(128,512,CHUNKING,"CHUNKING__31", 134217728));
	schl.addKernel(makeElasticKernel(192,64,CHUNKING,"CHUNKING__32", 67108864));
	schl.addKernel(makeElasticKernel(224,32,CHUNKING,"CHUNKING__33", 33554432));
	schl.addKernel(makeElasticKernel(256,128,CHUNKING,"CHUNKING__34", 33554432));
	schl.addKernel(makeElasticKernel(96,2,CHUNKING,"CHUNKING__35", 33554432));
	schl.addKernel(makeElasticKernel(192,2,CHUNKING,"CHUNKING__36", 33554432));
	schl.addKernel(makeElasticKernel(512,1,CHUNKING,"CHUNKING__37", 67108864));
	schl.addKernel(makeElasticKernel(160,32,CHUNKING,"CHUNKING__38", 67108864));
	schl.addKernel(makeElasticKernel(64,16,CHUNKING,"CHUNKING__39", 67108864));
	schl.addKernel(makeElasticKernel(64,4,CHUNKING,"CHUNKING__40", 33554432));
	schl.addKernel(makeElasticKernel(512,16,CHUNKING,"CHUNKING__41", 67108864));
	schl.addKernel(makeElasticKernel(128,512,CHUNKING,"CHUNKING__42", 33554432));
	schl.addKernel(makeElasticKernel(64,128,CHUNKING,"CHUNKING__43", 134217728));
	schl.addKernel(makeElasticKernel(96,8,CHUNKING,"CHUNKING__44", 33554432));
	schl.addKernel(makeElasticKernel(512,8,CHUNKING,"CHUNKING__45", 134217728));
	schl.addKernel(makeElasticKernel(96,32,CHUNKING,"CHUNKING__46", 33554432));
	schl.addKernel(makeElasticKernel(96,1,CHUNKING,"CHUNKING__47", 67108864));
	schl.addKernel(makeElasticKernel(256,64,CHUNKING,"CHUNKING__48", 134217728));
	schl.addKernel(makeElasticKernel(224,8,CHUNKING,"CHUNKING__49", 67108864));
	schl.addKernel(makeElasticKernel(64,128,CHUNKING,"CHUNKING__50", 134217728));
	schl.addKernel(makeElasticKernel(192,1,CHUNKING,"CHUNKING__51", 67108864));
	schl.addKernel(makeElasticKernel(64,256,CHUNKING,"CHUNKING__52", 134217728));
	schl.addKernel(makeElasticKernel(224,1,CHUNKING,"CHUNKING__53", 134217728));
	schl.addKernel(makeElasticKernel(64,32,CHUNKING,"CHUNKING__54", 67108864));
	schl.addKernel(makeElasticKernel(64,16,CHUNKING,"CHUNKING__55", 33554432));
	schl.addKernel(makeElasticKernel(160,256,CHUNKING,"CHUNKING__56", 33554432));
	schl.addKernel(makeElasticKernel(192,64,CHUNKING,"CHUNKING__57", 33554432));
	schl.addKernel(makeElasticKernel(192,32,CHUNKING,"CHUNKING__58", 33554432));
	schl.addKernel(makeElasticKernel(256,8,CHUNKING,"CHUNKING__59", 67108864));
	schl.addKernel(makeElasticKernel(128,4,CHUNKING,"CHUNKING__60", 67108864));
	schl.addKernel(makeElasticKernel(256,128,CHUNKING,"CHUNKING__61", 33554432));
	schl.addKernel(makeElasticKernel(128,1,CHUNKING,"CHUNKING__62", 33554432));
	schl.addKernel(makeElasticKernel(32,2,CHUNKING,"CHUNKING__63", 33554432));
	schl.addKernel(makeElasticKernel(32,64,CHUNKING,"CHUNKING__64", 67108864));
	schl.addKernel(makeElasticKernel(64,512,CHUNKING,"CHUNKING__65", 134217728));
	schl.addKernel(makeElasticKernel(64,32,CHUNKING,"CHUNKING__66", 134217728));
	schl.addKernel(makeElasticKernel(128,128,CHUNKING,"CHUNKING__67", 134217728));
	schl.addKernel(makeElasticKernel(256,2,CHUNKING,"CHUNKING__68", 134217728));
	schl.addKernel(makeElasticKernel(512,2,CHUNKING,"CHUNKING__69", 134217728));
	schl.addKernel(makeElasticKernel(96,512,CHUNKING,"CHUNKING__70", 134217728));
	schl.addKernel(makeElasticKernel(128,128,CHUNKING,"CHUNKING__71", 33554432));
	schl.addKernel(makeElasticKernel(160,1,CHUNKING,"CHUNKING__72", 67108864));
	schl.addKernel(makeElasticKernel(192,2,CHUNKING,"CHUNKING__73", 33554432));
	schl.addKernel(makeElasticKernel(160,16,CHUNKING,"CHUNKING__74", 33554432));
	schl.addKernel(makeElasticKernel(512,1,CHUNKING,"CHUNKING__75", 134217728));
	schl.addKernel(makeElasticKernel(32,32,CHUNKING,"CHUNKING__76", 67108864));
	schl.addKernel(makeElasticKernel(224,512,CHUNKING,"CHUNKING__77", 134217728));
	schl.addKernel(makeElasticKernel(192,1,CHUNKING,"CHUNKING__78", 33554432));
	schl.addKernel(makeElasticKernel(32,512,CHUNKING,"CHUNKING__79", 33554432));
	schl.addKernel(makeElasticKernel(64,256,CHUNKING,"CHUNKING__80", 33554432));
	schl.addKernel(makeElasticKernel(256,8,CHUNKING,"CHUNKING__81", 134217728));
	schl.addKernel(makeElasticKernel(64,512,CHUNKING,"CHUNKING__82", 33554432));
	schl.addKernel(makeElasticKernel(224,8,CHUNKING,"CHUNKING__83", 67108864));
	schl.addKernel(makeElasticKernel(224,512,CHUNKING,"CHUNKING__84", 33554432));
	schl.addKernel(makeElasticKernel(64,8,CHUNKING,"CHUNKING__85", 33554432));
	schl.addKernel(makeElasticKernel(128,64,CHUNKING,"CHUNKING__86", 134217728));
	schl.addKernel(makeElasticKernel(512,1,CHUNKING,"CHUNKING__87", 33554432));
	schl.addKernel(makeElasticKernel(256,128,CHUNKING,"CHUNKING__88", 33554432));
	schl.addKernel(makeElasticKernel(512,32,CHUNKING,"CHUNKING__89", 33554432));
	schl.addKernel(makeElasticKernel(128,16,CHUNKING,"CHUNKING__90", 67108864));
	schl.addKernel(makeElasticKernel(64,2,CHUNKING,"CHUNKING__91", 33554432));
	schl.addKernel(makeElasticKernel(96,4,CHUNKING,"CHUNKING__92", 134217728));
	schl.addKernel(makeElasticKernel(96,512,CHUNKING,"CHUNKING__93", 33554432));
	schl.addKernel(makeElasticKernel(256,4,CHUNKING,"CHUNKING__94", 33554432));
	schl.addKernel(makeElasticKernel(32,64,CHUNKING,"CHUNKING__95", 67108864));
	schl.addKernel(makeElasticKernel(192,2,CHUNKING,"CHUNKING__96", 67108864));
	schl.addKernel(makeElasticKernel(160,128,CHUNKING,"CHUNKING__97", 67108864));
	schl.addKernel(makeElasticKernel(256,1,CHUNKING,"CHUNKING__98", 33554432));
	schl.addKernel(makeElasticKernel(512,16,CHUNKING,"CHUNKING__99", 134217728));
	schl.addKernel(makeElasticKernel(512,8,CHUNKING,"CHUNKING__100", 33554432));
	schl.addKernel(makeElasticKernel(192,32,CHUNKING,"CHUNKING__101", 134217728));
	schl.addKernel(makeElasticKernel(128,4,CHUNKING,"CHUNKING__102", 33554432));
	schl.addKernel(makeElasticKernel(224,512,CHUNKING,"CHUNKING__103", 67108864));
	schl.addKernel(makeElasticKernel(192,128,CHUNKING,"CHUNKING__104", 33554432));
	schl.addKernel(makeElasticKernel(32,16,CHUNKING,"CHUNKING__105", 33554432));
	schl.addKernel(makeElasticKernel(32,1,CHUNKING,"CHUNKING__106", 67108864));
	schl.addKernel(makeElasticKernel(256,256,CHUNKING,"CHUNKING__107", 33554432));
	schl.addKernel(makeElasticKernel(128,512,CHUNKING,"CHUNKING__108", 134217728));
	schl.addKernel(makeElasticKernel(160,32,CHUNKING,"CHUNKING__109", 134217728));
	schl.addKernel(makeElasticKernel(224,256,CHUNKING,"CHUNKING__110", 67108864));
	schl.addKernel(makeElasticKernel(224,256,CHUNKING,"CHUNKING__111", 67108864));
	schl.addKernel(makeElasticKernel(128,512,CHUNKING,"CHUNKING__112", 33554432));
	schl.addKernel(makeElasticKernel(512,512,CHUNKING,"CHUNKING__113", 67108864));
	schl.addKernel(makeElasticKernel(224,128,CHUNKING,"CHUNKING__114", 33554432));
	schl.addKernel(makeElasticKernel(192,128,CHUNKING,"CHUNKING__115", 134217728));
	schl.addKernel(makeElasticKernel(96,16,CHUNKING,"CHUNKING__116", 134217728));
	schl.addKernel(makeElasticKernel(256,32,CHUNKING,"CHUNKING__117", 33554432));
	schl.addKernel(makeElasticKernel(512,64,CHUNKING,"CHUNKING__118", 33554432));
	schl.addKernel(makeElasticKernel(160,1,CHUNKING,"CHUNKING__119", 67108864));
	schl.addKernel(makeElasticKernel(160,64,CHUNKING,"CHUNKING__120", 67108864));
	schl.addKernel(makeElasticKernel(512,128,CHUNKING,"CHUNKING__121", 67108864));
	schl.addKernel(makeElasticKernel(512,8,CHUNKING,"CHUNKING__122", 33554432));
	schl.addKernel(makeElasticKernel(256,32,CHUNKING,"CHUNKING__123", 134217728));
	schl.addKernel(makeElasticKernel(224,128,CHUNKING,"CHUNKING__124", 67108864));
	schl.addKernel(makeElasticKernel(32,512,CHUNKING,"CHUNKING__125", 67108864));
	schl.addKernel(makeElasticKernel(224,16,CHUNKING,"CHUNKING__126", 33554432));
	schl.addKernel(makeElasticKernel(160,2,CHUNKING,"CHUNKING__127", 67108864));
	schl.addKernel(makeElasticKernel(256,32,CHUNKING,"CHUNKING__128", 67108864));
	schl.addKernel(makeElasticKernel(192,4,CHUNKING,"CHUNKING__129", 67108864));
	schl.addKernel(makeElasticKernel(128,1,CHUNKING,"CHUNKING__130", 33554432));
	schl.addKernel(makeElasticKernel(512,64,CHUNKING,"CHUNKING__131", 134217728));
	schl.addKernel(makeElasticKernel(64,512,CHUNKING,"CHUNKING__132", 134217728));
	schl.addKernel(makeElasticKernel(96,128,CHUNKING,"CHUNKING__133", 67108864));
	schl.addKernel(makeElasticKernel(96,2,CHUNKING,"CHUNKING__134", 67108864));
	schl.addKernel(makeElasticKernel(128,8,CHUNKING,"CHUNKING__135", 134217728));
	schl.addKernel(makeElasticKernel(64,64,CHUNKING,"CHUNKING__136", 67108864));
	schl.addKernel(makeElasticKernel(32,128,CHUNKING,"CHUNKING__137", 134217728));
	schl.addKernel(makeElasticKernel(128,1,CHUNKING,"CHUNKING__138", 67108864));
	schl.addKernel(makeElasticKernel(32,1,CHUNKING,"CHUNKING__139", 134217728));
	schl.addKernel(makeElasticKernel(160,4,CHUNKING,"CHUNKING__140", 33554432));


}

#endif /* WORKLOADGENERATION_H_ */
