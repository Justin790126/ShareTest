%%writefile print_kid.cu

#include<stdio.h>

__global__ void print_threadIds()
{
  printf("xid : %d, yid: %d, zid:%d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
  int nx,ny;
  nx = 16;
  ny = 16;
  dim3 block(8,8,1);
  dim3 grid(nx/block.x,ny/block.y,1);
  print_threadIds<<<grid,block>>>();
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return 0;
}