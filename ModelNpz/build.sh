#!/bin/bash

rm -rf build
mkdir build
cd build
cmake ../
make
cd ../

cp lena.png build/