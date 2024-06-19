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

#define N 131072

long count_key(long Nlen, long* a, long key)
{
    long count = 0;
    for (int i =0;i< Nlen; i++) {
        if (a[i] == key) count++;
    }
    return count;
}

long count_iter(long Nlen, long* a, long key)
{
    long count = 0;
    int last = -1;
    #pragma omp taskloop num_tasks(omp_get_num_threads()) reduction(+:count)
    for (int i =0;i< Nlen; i++) {
        if (omp_get_thread_num() != last)
            cout << omp_get_thread_num() << endl;
            last = omp_get_thread_num();
        if (a[i] == key) ++count;
    }
    return count;
}

#define CUTOFF 3

long count_recur(long Nlen, long* a, long key, int depth)
{
    long count = 0;
    long count1, count2;
    if (Nlen == 1) {
        if (a[0] == key) ++count;
    }
    else {
        if (depth < CUTOFF) {
            #pragma omp task shared(count1)
            count1 = count_recur(Nlen/2, a, key, depth);
            #pragma omp task shared(count2)
            count2 = count_recur(Nlen-Nlen/2, a+Nlen/2, key, depth);
            #pragma omp taskwait
        } else {
            count1 = count_recur(Nlen/2, a, key, depth);
            count2 = count_recur(Nlen-Nlen/2, a+Nlen/2, key, depth);
        }
    }
    depth++;
    return count1 + count2;
}

int main(int argc, char** argv) {
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    long a[N], key=42, nkey = 0;
    for (long i =0;i<N;i++) a[i] = random()%N;
    a[N%43]=key;a[N%73]=key;a[N%3]=key;
    auto start = std::chrono::high_resolution_clock::now();

    int mode = stoi(argv[1]);
    if (mode == 0) nkey = count_key(N, a, key);
    else if (mode == 1) {
        #pragma omp parallel
        #pragma omp single
        nkey = count_iter(N, a, key);
    } else if (mode == 2) {
        int dep = 0;
        #pragma omp parallel
        #pragma omp single
        nkey = count_recur(N, a, key, dep);
    }
    
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "nkey: " << nkey << endl;
    // Print the execution time
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    
    

    return 0;
}