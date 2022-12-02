#!/bin/bash

nodename=icx1      # Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
#nodename=spr       # cerberos spr2, July 29
#nodename=clx        # Intel(R) Xeon(R) Platinum 8260L CPU @ 2.40GHz
#nodename=atsb0		# Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz
#nodename=sprh

NEV=64
NMOM=33

ListNX=(16 32 64 96)
ListND=(16 32 64 96)

result_file="measurements/perf_${nodename}_variousXD_nEV${NEV}_nMom${nMom}.txt"
binary=kernel_$nodename
tmp=tmp_$nodename.txt

printf "%s \t %s \t %s \t  %s \t %s \t  %s \n" "nX" "nD" "Apply" "Reconstruct" "ParComp" "Performance"  >> $result_file
printf "%s \t %s \t %s \t  %s \t %s \t  %s \n" "nX" "nD" "Apply" "Reconstruct" "ParComp" "Performance"

for i in "${!ListNX[@]}"; do
    for j in "${!ListND[@]}"; do
	    export nX="${ListNX[i]}"
	    export nD="${ListND[j]}"
        make clean 2>&1 >/dev/null && make 2>&1 > /dev/null
        cp kernel $binary
	    OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close ./$binary > $tmp
	    times=$(tail -n 6 $tmp | grep -oE "[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?." | tr '\n' '\t')
	    printf "%s \t %s \t %s \t  %s \t %s \t %s \n" "$nX" "$nD" "$times"  >> $result_file
	    printf "%s \t %s \t %s \t  %s \t %s \t %s \n" "$nX" "$nD" "$times"
	    rm $tmp
    done
done

rm $binary
