#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>

class SharedMemory {
public:
    SharedMemory(const char* name, size_t size) {
        // Create a shared memory object
        m_sName = name;
        shm_fd = shm_open(m_sName, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            perror("shm_open");
            exit(1);
        }

        // Set the size of the shared memory object
        if (ftruncate(shm_fd, size) == -1) {
            perror("ftruncate");
            exit(1);
        }

        // Map the shared memory object into the process's address space
        ptr = (char*)mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (ptr == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }
    }

    ~SharedMemory() {
        // Unmap the shared memory object
        if (munmap(ptr, size) == -1) {
            perror("munmap");
            exit(1);
        }

        // Close the shared memory object
        if (close(shm_fd) == -1) {
            perror("close");
            exit(1);
        }

        // Remove the shared memory object
        if (shm_unlink(m_sName) == -1) {
            perror("shm_unlink");
            exit(1);
        }
    }

    float* getFloatPtr() {
        return reinterpret_cast<float*>(ptr);
    }

private:
    int shm_fd;
    char* ptr;
    size_t size;
    const char* m_sName;
};

