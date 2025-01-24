#!/bin/bash

SRC_DIR="/home/justin126/workspace/ShareTest/prototest"
DST_DIR="/home/justin126/workspace/ShareTest/prototest/include"
SRC_DIR1="/home/justin126/workspace/venv/lib/python3.7/site-packages/tensorflow/include/tensorflow/core/util/"
SRC_DIR2="/home/justin126/workspace/venv/lib/python3.7/site-packages/tensorflow/include/tensorflow/core/framework/"
protoc -I=${SRC_DIR1} -I=${SRC_DIR2} --cpp_out=${DST_DIR} ${SRC_DIR1}/event.proto ${SRC_DIR2}/summary.proto