#!/bin/bash

foldname=normImage
aulist=(1 2 4 6 7 10 12 14 15 17 23)
run=$2
dir=/home/handong/Shizhong/BP4D/data_steps/
datadir=$dir/DRMF_CUDA/
#for run in 1 2 3 4 5 #6 7 8 9 10
#do
    resdir=$dir/testResult/${foldname}_adaptconv3one444_run${run}/
    mkdir $resdir
    for iau in 0 1 2 3 4 5 6 7 8 9 10 
    do
    curAU=${aulist[${iau}]}
    scorefile=$resdir/outscore_baseline${curAU}.txt
    rm $scorefile
    if [ ! -e "$scorefile" ]
    then
    	echo "$scorefile not exist"
    	./models/bp4d/solver_train_adaptconv.sh ${curAU} ${datadir} ${resdir} $foldname
    	./build/tools/caffe train -gpu $1 -solver=$resdir/solver${curAU}.prototxt  #-weights=/media/datadisk/database/FERA2015/data_steps/CNNData/normWarpImage/AU_binary_fromFera2015.caffemodel
    fi
    done
#done
