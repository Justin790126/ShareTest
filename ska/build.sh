#!/bin/bash

rm -rf build
mkdir build
cp lena4096_4096.png ./build/
cd build
cmake ../
make
cd ../
