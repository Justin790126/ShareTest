#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../
make
cd ../

# mpirun -np 4 ./build/m1
# mpirun -np 2 ./build/m2
