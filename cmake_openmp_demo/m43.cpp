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
    #pragma omp taskloop grainsize(4)
    for (size_t i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
        printf("[%d]->%d\n", who, i);
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
    #pragma omp single
    vector_add(a,b,c,n);

    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cout << omp_get_max_threads() << endl;
    // Print the execution time
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    // for (size_t i = 0; i < 10; i++)
    // {
    //     cout << c[i] << ", ";
    // }
    cout << endl;
    

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}