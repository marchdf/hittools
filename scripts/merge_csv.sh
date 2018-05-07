#!/bin/bash
#
# Merge CSV files together

res=${1}
nprocs=${2}
pfx='fv'

tmpname="merged_${res}.dat"
head -1 ${pfx}_${res}_${nprocs}_0.dat > ${tmpname}
for ((rank=0;rank<${nprocs};++rank))
do
    tail -n +2 -q ${pfx}_${res}_${nprocs}_${rank}.dat >> ${tmpname}
done
