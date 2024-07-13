%%writefile dev.cu

#include<stdio.h>

void query_device()
{
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    printf("No cuda support\n");
  }

  int devNo = 0;
  cudaDeviceProp iProp;
  cudaGetDeviceProperties(&iProp, devNo);

  printf("Device %d: %s\n", devNo, iProp.name);
  printf(".   Number of multiprocessors:          %d\n",
    iProp.multiProcessorCount);
  printf(".   clock rate:                     %d\n",
    iProp.clockRate);
  printf(".   Compute capability :        %d.%d\n",
    iProp.major, iProp.minor);
  printf(".   Total amount of global mem:       %4.2f KB\n",
    iProp.totalGlobalMem/1024.0);
  printf(".   Total amount of constant mem:       %4.2f KB\n",
    iProp.totalConstMem/1024.0);
  printf(".   Memory Clock Rate:               %d\n",
    iProp.memoryClockRate);
  printf(".   Maximum memory length:               %d\n",
    iProp.memoryClockRate);
  printf(".   Maximum threads per multiprocessor: %d\n",
    iProp.maxThreadsPerMultiProcessor);
  printf(".   Maximum threads per block:          %d\n",
    iProp.maxThreadsPerBlock);
  printf(".   Maximum threads per grid:            %d\n",
    iProp.maxThreadsDim[0]*iProp.maxThreadsDim[1]*iProp.maxThreadsDim[2]);
  printf(".   Maximum block size:                  %d\n",
    iProp.maxThreadsDim[0]*iProp.maxThreadsDim[1]*iProp.maxThreadsDim[2]);
  printf(".   Maximum grid size:                   %d\n",
    iProp.maxGridSize[0]*iProp.maxGridSize[1]*iProp.maxGridSize[2]);
  
}

int main()
{
  query_device();
  return 0;
}