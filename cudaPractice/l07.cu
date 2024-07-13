%%writefile h2d.cu

#include<stdio.h>


__global__ void mem_trs_test(int* data)
{
  int gid = blockDim.x*blockIdx.x + threadIdx.x;
  printf("tid: %d, gid: %d, value:%d\n", threadIdx.x, gid, data[gid]);
}

__global__ void mem_trs_test2(int* data, int size)
{
  int gid = blockDim.x*blockIdx.x + threadIdx.x;
  if (gid < size)
    printf("tid: %d, gid: %d, value:%d\n", threadIdx.x, gid, data[gid]);
}


int main()
{
  int size = 150;
  int byte_size = size*sizeof(int);

  int* input = (int*)malloc(byte_size);

  time_t t;
  srand((unsigned)time(&t));
  for(int i = 0; i < size; i++)
  {
    input[i] = (int)(rand()&0xff);
  }

  int* d_input;
  cudaMalloc((void**)&d_input, byte_size);
  cudaMemcpy(d_input, input, byte_size, cudaMemcpyHostToDevice);

  dim3 block(32);
  dim3 grid(5);

  mem_trs_test<<<grid, block>>>(d_input, size);
  cudaDeviceSynchronize();

  cudaFree(d_input);
  free(input);

  cudaDeviceReset();


  return 0;
}