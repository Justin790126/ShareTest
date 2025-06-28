#!/bin/bash

FILE=$1

NUM_OF_SPLIT=3

xxd -p "$FILE" > all.hex
split -n $NUM_OF_SPLIT all.hex hexpart_

