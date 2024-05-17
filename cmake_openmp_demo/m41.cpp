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

void vector_add(int* A, int* B, int* C, int n) {
    int who = omp_get_thread_num();
    int nt = omp_get_num_threads();
    int BS = n/nt;
    printf("[%d]\n", who);
    for (size_t i = who*BS; i < (who+1)*BS; i++)
    {
        C[i] = A[i] + B[i];
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
    #pragma omp parallel
    vector_add(a,b,c,n);
    
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    for (size_t i = 0; i < n; i++)
    {
        cout << c[i] << ", ";
    }
    cout << endl;
    

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}