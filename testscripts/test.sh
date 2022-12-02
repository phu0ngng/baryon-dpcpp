#!/bin/bash

result_file=result.txt

#ListOMP=(2 4 8 16 32 64)
ListOMP=(8 16)
#ListOMP=(12 24 48)
#touch $result_file
printf "%s \t %s \t %s \t  %s \t  %s \t  %s \t  %s \n" "nOMP" "Apply" "Reconstruct" "CrossX" "ScalarX" "MomProj" "Performance"  >> $result_file
#export I_MPI_PIN_CELL=core

for i in "${!ListOMP[@]}"; do
	nOMP="${ListOMP[i]}"
	OMP_NUM_THREADS=$nOMP OMP_PLACES=cores OMP_PROC_BIND=close ./kernel 64 64 256 8 16 16 > tmp.txt
	times=$(cat tmp.txt | grep -oE "[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?." | sed -n '3,8p' | tr '\n' '\t')
	printf "%s \t %s \t  %s \t %s \t %s \t %s \t %s \n" "$nOMP" "$times"  >> $result_file
	printf "%s \t %s \t  %s \t %s \t %s \t %s \t %s \n" "$nOMP" "$times"
	rm tmp.txt
done

cat $result_file
