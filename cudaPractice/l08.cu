%%writefile sum.cu

#include<stdio.h>


__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if (gid < size) {
    c[gid] = a[gid] + b[gid];
  }
}

void sum_array_cpu(int* a, int* b, int* c, int size)
{
  for(int i = 0; i < size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

void cmp_array(int* a, int* b, int size)
{
  for(int i =0; i < size; i++)
  {
    if (a[i] != b[i]) {
      printf("Array are different\n");
      return;
    }
  }
  printf("Array are same\n");
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

int main()
{
  int size = 10000;
  int block_size = 128;
  cudaError error;

  int NO_BYTES = size*sizeof(int);

  int* h_a, *h_b, *gpu_results, *h_c;

  h_a = (int*)malloc(NO_BYTES);
  h_b = (int*)malloc(NO_BYTES);
  h_c = (int*)malloc(NO_BYTES);
  gpu_results = (int*)malloc(NO_BYTES);

  time_t t;
  srand((unsigned)time(&t));
  for(int i = 0; i < size; i++)
  {
    h_a[i] = (int)(rand()&0xff);
  }

  for(int i = 0; i < size; i++)
  {
    h_b[i] = (int)(rand()&0xff);
  }

  sum_array_cpu(h_a, h_b, h_c, size);

  memset(gpu_results,0,NO_BYTES);

  int* d_a, *d_b, *d_c;
  gpuErrchk(cudaMalloc((int**)&d_a, NO_BYTES));
  
  gpuErrchk(cudaMalloc((int**)&d_b, NO_BYTES));
 
  gpuErrchk(cudaMalloc((int**)&d_c, NO_BYTES));
 
  cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
  
  dim3 block(block_size);
  dim3 grid((size/block.x)+1);

  sum_array_gpu<<<grid,block>>>(d_a, d_b, d_c, size);
  cudaDeviceSynchronize();

  cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);


  cmp_array(gpu_results, h_c, size);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(gpu_results);

  cudaDeviceReset();

  return 0;
}