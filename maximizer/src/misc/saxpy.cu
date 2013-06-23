#include "cuda_launch_config.hpp"
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <cassert>
#include "stdio.h"
#include "occupancy_tools/OccupancyCalculator.hpp"
//#include "Maximizer.hpp"
#include "occupancy_tools/OccupancyLimits.hpp"
__global__ void saxpy(float a, float *x, float *y, size_t n)
{

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
  {
    y[i] = a * x[i] + y[i];
  }
}

int main()
{
  size_t n = 1 << 20;;
  thrust::device_vector<float> x(n, 10);
  thrust::device_vector<float> y(n, 100);

  float a = 10;

  // we'd like to launch saxpy, but we're not sure which block size to use
  // let's use a heuristic which promotes occupancy.

  // first, get the cudaFuncAtttributes object corresponding to saxpy
  cudaFuncAttributes attributes;
  cudaFuncGetAttributes(&attributes, saxpy);

  // next, get the cudaDeviceProp object corresponding to the current device
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  // we can combine the two to compute a block size
 // printf("%d\n", __cuda_launch_config_detail::max_active_blocks_per_multiprocessor(properties,attributes,512,0));
  //printf("%d\n", getMaxActiveBlocksPerSM(properties,attributes,512));

  attributes.numRegs = 200;
  BlockUsage usage = getBlockUsageStats(properties,attributes,1);
  printf("+++++++++++++Kernel Stats+++++++++++++\n");

  printf("Regs per thread %d\n", attributes.numRegs);


  printf("+++++++++++++Calculation output+++++++++++++\n");
  printf("Blocks %d\n", usage.blocksPerSM);
  printf("Threads %d\n", usage.numThreads);
  printf("Registers %d\n", usage.numRegisters);
  printf("SMemory %d\n", usage.sharedMemory);
  printf("Blocks %d\n", __cuda_launch_config_detail::max_active_blocks_per_multiprocessor(properties,attributes,1,0));



  // compute the number of blocks of size num_threads to launch
  //size_t num_blocks = n / num_threads;

  // check for partial block at the end
/*
  if(n % num_threads) ++num_blocks;

  saxpy<<<num_blocks,num_threads>>>(a, raw_pointer_cast(x.data()), raw_pointer_cast(y.data()), n);
*/

  // validate the result
  //assert(thrust::all_of(y.begin(), y.end(), thrust::placeholders::_1 == 200));

  return 0;
}

