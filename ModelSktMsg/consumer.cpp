
// Consumer.cpp
#include "SharedMemory.h"

int main() {
    SharedMemory shm("my_shared_memory", 4096 * 4096 * sizeof(float));
    float* ptr = shm.getFloatPtr();

    // Access and process the data from shared memory
    for (int i = 0; i < 4096 * 4096; ++i) {
        printf("Value at index %d: %f\n", i, ptr[i]);
    }

    return 0;
}