%%writefile print_gid.cu

#include<stdio.h>

__global__ void print_detail()
{
  printf("blockId.x: %d, blockId.y: %d, blockId.z: %d,\
          blockDim.x: %d, blockDim.y: %d\
          gridDim.x: %d, gridDim.y: %d \n",
          blockIdx.x, blockIdx.y, blockIdx.z,
          blockDim.x, blockDim.y,
          gridDim.x, gridDim.y);
}

int main()
{
  int nx,ny;
  nx = 16;
  ny = 16;
  dim3 block(8,8,1);
  dim3 grid(nx/block.x,ny/block.y,1);
  print_detail<<<grid,block>>>();
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return 0;
}