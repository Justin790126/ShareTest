#!/bin/bash

rm -rf build
mkdir build
cp test.xml ./build/
cd build
cmake ../
make
cd ../
