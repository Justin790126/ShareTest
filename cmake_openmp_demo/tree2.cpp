#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

/*
    Implicit task
*/

#define N 1024
#define MIN_SIZE 64
int result = 0;

int do_product(int* A, int* B , int n)
{
    int tmp = 0;
    for (size_t i = 0; i < n; i++)
    {
        tmp += A[i]*B[i];
    }
    return tmp;
}

int rec_dot_product(int* A, int* B, int n)
{
    int tmp1, tmp2 = 0;
    if (n > MIN_SIZE) {
        int n2 = n/2;
        tmp1 = rec_dot_product(A,B, n2);
        tmp2 = rec_dot_product(A+n2, B+n2, n-n2);
    } else {
        tmp1 = do_product(A,B,n);
    }
    return tmp1+tmp2;
}

int rec_dot_productMT1(int* A, int* B, int n)
{
    int tmp1, tmp2 = 0;
    if (n > MIN_SIZE) {
        int n2 = n/2;
        #pragma omp task shared(tmp1)
        tmp1 = rec_dot_productMT1(A,B, n2);
        #pragma omp task shared(tmp2)
        tmp2 = rec_dot_productMT1(A+n2, B+n2, n-n2);
        #pragma omp taskwait
    } else {
        tmp1 = do_product(A,B,n);
    }
    return tmp1+tmp2;
}
#define CUTOFF 3
int rec_dot_productMT2(int* A, int* B, int n, int depth)
{
    int tmp1, tmp2 = 0;
    if (n > MIN_SIZE) {
        int n2 = n/2;
        if (!omp_in_final()) {
            #pragma omp task shared(tmp1) final(depth >= CUTOFF)
            tmp1 = rec_dot_productMT2(A,B, n2, depth+1);
            #pragma omp task shared(tmp2) final(depth >= CUTOFF)
            tmp2 = rec_dot_productMT2(A+n2, B+n2, n-n2, depth+1);
            #pragma omp taskwait
        } else {
            tmp1 = rec_dot_productMT2(A,B, n2, depth+1);
            tmp2 = rec_dot_productMT2(A+n2, B+n2, n-n2, depth+1);
        }
        
    } else {
        tmp1 = do_product(A,B,n);
    }
    return tmp1+tmp2;
}

int main(int argc, char** argv) {
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    int n = stoi(argv[1]);
    cout << "n: " << n << endl;
    int* a = new int[n];
    int* b = new int[n];
    int* c = new int[n];
    for (size_t i = 0; i < n; i++)
    {
        a[i] = b[i] = i;
        c[i] = 0;
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    int mode = stoi(argv[2]);
    if (mode == 0) {
        result = rec_dot_product(a,b,n);
    } else if (mode == 1) {
        #pragma omp parallel
        #pragma omp single
        result = rec_dot_product(a,b,n);
    } else if (mode == 2) {
        int level = 0;
        #pragma omp parallel
        #pragma omp single
        rec_dot_productMT2(a,b,n, level);
    }
    // 
    

    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    
    cout << "result: " << result << endl;
    

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}