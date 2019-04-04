#!/bin/sh
./sssp $1 -j out.json --with-header --device=0 --src=0 --configs=$2 --verbose --validation
