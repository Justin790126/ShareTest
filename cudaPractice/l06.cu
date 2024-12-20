%%writefile idx_cal2d_2d.cu

#include<stdio.h>


__global__ void unique_gid_calculation_2d_2d(int* data)
{
  int tid = blockDim.x*threadIdx.y + threadIdx.x;
  int num_threads_in_a_block = blockDim.x*blockDim.y;
  int block_offset = num_threads_in_a_block*blockIdx.x;

  int num_threads_in_a_row = num_threads_in_a_block*gridDim.x;
  int row_offset = num_threads_in_a_row*blockIdx.y;

  int gid = tid+block_offset+row_offset;
  printf("blockIdx: %d, blockIdx.y: %d tid: %d, gid: %d, data: %d\n",
  blockIdx.x, blockIdx.y, tid, gid, data[gid]);
}

int main()
{
  int arr_size = 16;
  int array_byte_size = sizeof(int) * arr_size;
  int h_data[] = {23,9,4,53,65,12,1,33,87,45,23,12,342,56,44,99};
  for(int i = 0; i < arr_size; i++)
  {
    printf("h_data[%d]: %d\n", i, h_data[i]);
  }
  printf("\n \n");

  int* d_data;
  cudaMalloc((void**)&d_data, array_byte_size);
  cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

  dim3 block(2,2);
  dim3 grid(2,2);

  unique_gid_calculation_2d_2d<<<grid, block>>>(d_data);
  cudaDeviceSynchronize();

  cudaDeviceReset();


  return 0;
}