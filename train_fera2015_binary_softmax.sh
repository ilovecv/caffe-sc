#!/usr/bin/env sh

foldname=$1
k=$2
aulist=(2 12 17 25 28 45)
fullnode=(64 128 256 512 1024 2048)
num=${#fullnode[@]}
#for (( k=0; k<2; k++ ))
#do 
    for j in 1 2 3 4 5 6 7 8 9 10
    do
        mkdir models/fera2015/${foldname}_binarysm${j}_fc${fullnode[$k]}/
        mkdir /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}_binarysm${j}_fc${fullnode[$k]}/
        rm /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}_binarysm${j}_fc${fullnode[$k]}/*
        for i in 0 1 2 3 4 5
        do
        curAU=${aulist[${i}]}
        ./models/fera2015/solver_train_binary_softmax.sh ${foldname} ${curAU} ${j} ${fullnode[$k]}
        ./build/tools/caffe train -gpu 0 -solver=models/fera2015/${foldname}_binarysm${j}_fc${fullnode[$k]}/solver${curAU}.prototxt #-weights=/media/datadisk/database/FERA2015/data_steps/CNNData/normWarpImage/AU_binarysm_fromFera2015.caffemodel
        done
    done
#done
