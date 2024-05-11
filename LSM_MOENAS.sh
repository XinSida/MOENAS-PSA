#!/bin/bash

echo '#: Input the number of training examples: '
read num_example

using_resting=1

python "MOENAS.py" $num_example $using_resting 330 170 

echo ''Search has been finished!'' 
echo '#----------------------------------------#'
