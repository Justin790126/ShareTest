#!/bin/bash

rm -rf build
mkdir build
cp rcsv.txt ./build/
cd build
cmake ../
make
cd ../
