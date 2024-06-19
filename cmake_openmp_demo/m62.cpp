#include <omp.h>
#include <vector>
#include <chrono>
#include <iostream>

int main() {
    const int N = 1000000;
    std::vector<int> vec(N, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        #pragma omp critical
        {
            vec[i] += 1;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time with #pragma omp critical: " << duration.count() << "s\n";

    omp_lock_t lock;
    omp_init_lock(&lock);

    start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        omp_set_lock(&lock);
        vec[i] += 1;
        omp_unset_lock(&lock);
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time with omp_set_lock: " << duration.count() << "s\n";

    omp_destroy_lock(&lock);
    
    return 0;
}
