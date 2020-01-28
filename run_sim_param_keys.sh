#!/bin/bash

# Author: Prannath Moolchand
# Date: Jan 15, 2020
# A script to generate hdf5 simulations from keys in param_file passed to nested
# Takes main config file and param file as

config_fil=$1
param_fil=$2

#echo $config_fil
#echo $param_fil

delimiter="param_file_"
s=$param_fil$delimiter
array=();
while [[ $s ]]; do
    array+=( "${s%%"$delimiter"*}" );
    s=${s#*"$delimiter"};
done;

sim_id_temp=${array[1]}
sim_id=${sim_id_temp%.*}

keys=$(grep "^[^ ]" $param_fil) 

for var in $keys
do
    short_key=(${var//_specialist/ })
    output_fil="data/$sim_id""_$short_key.hdf5"
    mpirun -n 6 python3 -m nested.optimize --config-file-path=$config_fil --analyze --export --param_file=$param_fil --x0_key=$var --export-file-path=$output_fil --label=$var
    echo $var
done

