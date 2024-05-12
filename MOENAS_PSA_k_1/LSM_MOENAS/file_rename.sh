#!/bin/bash

file_name_rename=`ls -l ./MOENAS_PSA_k_1/LSM_MOENAS/output/results/ | awk -F [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9] '{print $1}' | awk -F :[0-9][0-9]\ _ '{print $2}' | sed s/-//g` 
file_name_rename_arr=($file_name_rename)
rename_length=${#file_name_rename_arr[*]} 
result_file_name=`ls -l ./MOENAS_PSA_k_1/LSM_MOENAS/output/results/ | awk -F :[0-9][0-9]\  '{print $2}'` 
result_file_name_arr=($result_file_name)
arr_length=${#result_file_name_arr[*]}
arr_index_last=`expr ${arr_length} - 1`
unset result_file_name_arr[$arr_index_last]      
arr_length=${#result_file_name_arr[*]}
k=0
for i in ${result_file_name_arr[*]}
do
    if [[ $i =~ "dynamic_array_spikemonitor" ]]
    then
        if [[ $i =~ "dynamic_array_spikemonitor_t_" ]]
        then
            cp "./MOENAS_PSA_k_1/LSM_MOENAS/output/results/${i}" "./MOENAS_PSA_k_1/LSM_MOENAS/result/dynamic_array_spikemonitor_t_"
        else
            cp "./MOENAS_PSA_k_1/LSM_MOENAS/output/results/${i}" `echo "./MOENAS_PSA_k_1/LSM_MOENAS/result/${file_name_rename_arr[$k]}" | sed -r 's/.{10}$//'`
        fi
    fi
    ((k=${k}+1))
done

