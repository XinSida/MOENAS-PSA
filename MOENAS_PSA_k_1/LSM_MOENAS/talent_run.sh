#!/bin/bash

echo "The number of training examples: $1"
echo "The number of testing examples: 10000"
bash_path='./MOENAS_PSA_k_1/LSM_MOENAS'
archive_path='./MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive'
archive_folder="${archive_path}"
if [ ! -d "${archive_folder}" ] ; then
    mkdir ${archive_folder}
fi
random_connection_folder="$archive_folder/random_connection"
result_folder_training="$archive_folder/result_training"
result_folder_testing="$archive_folder/result_testing"
weight_learned="$archive_folder/weight_learned"
if [ ! -d "${random_connection_folder}" ] ; then
    mkdir ${random_connection_folder}
    mkdir ${result_folder_training}
    mkdir ${result_folder_testing}
    mkdir ${weight_learned}
fi

IN_population=1156

python_script_name='LSM_simulation.py'

echo '#--------------------------------------------------------------------#'
echo "###: run the simulation iteration: training"

python "$bash_path/$python_script_name" 0 $1 $IN_population 330 170 
./MOENAS_PSA_k_1/LSM_MOENAS/file_rename.sh
mv ./MOENAS_PSA_k_1/LSM_MOENAS/result/* $result_folder_training

echo '#--------------------------------------------------------------------#'
echo "###: run the simulation iteration: testing"

python "$bash_path/$python_script_name" 1 10000 $IN_population 330 170
./MOENAS_PSA_k_1/LSM_MOENAS/file_rename.sh
mv ./MOENAS_PSA_k_1/LSM_MOENAS/result/* $result_folder_testing
