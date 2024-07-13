
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
