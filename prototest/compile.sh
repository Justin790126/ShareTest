#!/bin/bash

SRC_DIR="/home/justin126/workspace/ShareTest/prototest"
DST_DIR="/home/justin126/workspace/ShareTest/prototest/include"

protoc -I=${SRC_DIR} --cpp_out=${DST_DIR} ${SRC_DIR}/addressbook.proto