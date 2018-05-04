#!/bin/bash
#
# Merge binary files together

res=${1}
nprocs=${2}
pfx='fv'

tmpname="merged_${res}.in"
cat "${pfx}_${res}_${nprocs}_"*".in" > ${tmpname}
