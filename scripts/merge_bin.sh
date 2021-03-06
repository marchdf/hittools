#!/bin/bash
#
# Merge binary files together

res=${1}
nprocs=${2}
pfx='fv'

# Get sorted list of filenames
list=()
for ((rank=0;rank<${nprocs};++rank))
do
    list+=("${pfx}_${res}_${nprocs}_${rank}.in")
done

# Perform merge
oname="hit_ic_ut_${res}.in"
cat "${list[@]}" > ${oname}

