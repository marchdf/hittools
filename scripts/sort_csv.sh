#!/bin/bash
#
# Merge CSV files together

res=${1}
tmpname="merged_${res}.dat"

# Sort the csv file
oname="hit_ic_ut_${res}.dat"
head -n 1 ${tmpname} > ${oname}
tail -n +2 -q ${tmpname} | sort -k3 -k2 -k1 -g -t, >> ${oname}
