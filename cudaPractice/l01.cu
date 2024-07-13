%%writefile hello_cuda.cu

#include<stdio.h>

__global__ void hello_cuda()
{
  printf("hello CUDA world\n");
}

int main()
{
  int nx,ny;
  nx = 16;
  ny = 4;
  dim3 block(8,2,1);
  dim3 grid(nx/block.x,ny/block.y,1);
  hello_cuda<<<grid,block>>>();
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return 0;
}