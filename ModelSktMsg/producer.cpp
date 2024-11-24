// Producer.cpp
#include "SharedMemory.h"

int main() {
    SharedMemory shm("my_shared_memory", 4096 * 4096 * sizeof(float));
    float* ptr = shm.getFloatPtr();

    // Populate the shared memory with data
    for (int i = 0; i < 4096 * 4096; ++i) {
        ptr[i] = static_cast<float>(i); // Example data
    }

    while(true);
    return 0;
}
