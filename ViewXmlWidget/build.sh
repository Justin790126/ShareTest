#!/bin/bash

rm -rf build
mkdir build
cp test.xml ./build/
cp test2.xml ./build/
cd build
cmake ../
make
cd ../
