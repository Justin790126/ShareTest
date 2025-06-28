#!/bin/bash

FILE=$1

cat hexpart_* > all_recombined.hex
xxd -r -p all_recombined.hex $FILE
