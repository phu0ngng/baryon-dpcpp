#!/bin/bash

nodename=pvc      # Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

ListWS=(64 128 256 512 1024)
ListD2=(2 4 8 16)
ListD3=(16 32)

result_file="perf_${nodename}_scanning_blocksizes.txt"
binary=kernel_$nodename
tmp=tmp_$nodename.txt

mintime=100000000000
mincase=""

printf "%s \t %s \t %s \t  %s \t %s \n" "WS" "BD1" "BD2" "BD3" "Time"  >> $result_file
printf "%s \t %s \t %s \t  %s \t %s \n" "WS" "BD1" "BD2" "BD3" "Time"

for D3 in "${ListD3[@]}"; do
    for WS in "${ListWS[@]}"; do
        for D2 in "${ListD2[@]}"; do
            D1=$(($WS/($D2*$D3)))
            if [ $D1 -ge 1 ]; then
                sed -i "s/DPCPP_CXXFLAGS+= -DBLOCK_D1.*/DPCPP_CXXFLAGS+= -DBLOCK_D1=${D1} -DBLOCK_D2=${D2} -DBLOCK_D3=${D3}/" Makefile
                make clean 2>&1 >/dev/null && make 2>&1 > /dev/null
                cp kernel $binary
                ZE_AFFINITY_MASK=0.0 ./$binary > $tmp
                runtime=$(grep 'Average' $tmp | grep -oE "[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.")
                printf "%s \t %s \t %s \t  %s \t %s \n" "$WS" "$D1" "$D2" "$D3" "$runtime"  >> $result_file
                printf "%s \t %s \t %s \t  %s \t %s \n" "$WS" "$D1" "$D2" "$D3" "$runtime"
                if (( $(echo "$runtime < $mintime" |bc -l) )); then
                    mincase="$WS $D1 $D2 $D3"  
                    mintime=$runtime
                fi
                cat $tmp >> $result_file.log
                rm $tmp
            fi
        done
    done
done

echo "---------------------------------------"
printf "OPTIMAL CHOICE: WS=%s BlockD1=%s BlockD2=%s BlockD3=%s Runtime=%s \n" $mincase $mintime >> $result_file
printf "OPTIMAL CHOICE: WS=%s BlockD1=%s BlockD2=%s BlockD3=%s Runtime=%s \n" $mincase $mintime
rm $binary
