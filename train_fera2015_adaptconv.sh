#!/bin/bash

foldname=cnormWarpImage
aulist=(2 12 17 25 28 45)
dir=/home/handong/Shizhong/FERA2015/data_steps/
datadir=$dir/CNNData/${foldname}
for run in 1 #2 3 4 5 6 7 8 9 10
do
    resdir=$dir/testResult/${foldname}_adaptconv_run${run}/
    mkdir $resdir
    for i in 0 #1 2 3 4 5 
    do
    curAU=${aulist[${i}]}
    rm $resdir/outscore_baseline${curAU}.txt
    ./models/fera2015/solver_train_adaptconv.sh ${curAU} ${datadir} ${resdir}
    ./build/tools/caffe train -gpu $1 -solver=$resdir/solver${curAU}.prototxt  #-weights=/media/datadisk/database/FERA2015/data_steps/CNNData/normWarpImage/AU_binary_fromFera2015.caffemodel
    done
done
