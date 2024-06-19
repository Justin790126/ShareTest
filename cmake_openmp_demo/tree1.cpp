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

void do_product(int* A, int* B , int n)
{
    for (size_t i = 0; i < n; i++)
    {
        #pragma omp atomic
        result += A[i]*B[i];
    }
}

void rec_dot_product(int* A, int* B, int n)
{
    if (n > MIN_SIZE) {
        int n2 = n/2;
        rec_dot_product(A, B, n2);
        rec_dot_product(A+n2, B+n2, n-n2);
    } else {
        do_product(A,B,n);
    }
}

void rec_dot_productMT1(int* A, int* B, int n)
{
    if (n > MIN_SIZE) {
        int n2 = n/2;
        rec_dot_productMT1(A, B, n2);
        rec_dot_productMT1(A+n2, B+n2, n-n2);
    } else {
        #pragma omp task
        do_product(A,B,n);
    }
}

#define CUTOFF 2
void rec_dot_productMT2(int* A, int* B, int n, int depth)
{
    if (n > MIN_SIZE) {
        int n2 = n/2;
        if (depth == CUTOFF) {
            #pragma omp task
            {
                rec_dot_productMT2(A, B, n2, depth+1);
                rec_dot_productMT2(A+n2, B+n2, n-n2, depth+1);
            }
            
        } else {
            rec_dot_productMT2(A, B, n2, depth+1);
            rec_dot_productMT2(A+n2, B+n2, n-n2, depth+1);
        }
        
    } else {
        if (depth <= CUTOFF)
        {
            #pragma omp task
            do_product(A,B,n);
        } else {
            do_product(A,B,n);
        }
        
    }
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
        rec_dot_product(a,b,n);
    } else if (mode == 1) {
        #pragma omp parallel
        #pragma omp single
        rec_dot_productMT1(a,b,n);
    } else if (mode == 2) {
        int depth = 0;
        #pragma omp parallel
        #pragma omp single
        rec_dot_productMT2(a,b,n, depth);
    }
    // 
    

    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << omp_get_max_threads() << endl;
    // Print the execution time
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    
    cout << "result: " << result << endl;
    

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}