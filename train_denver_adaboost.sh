#!/usr/bin/env sh

foldname=$1
k=$2
aulist=(1 2 4 5 6 9 12 15 17 20 25 26)
set=(1 2 3 4 5)
#for (( k=1; k<3; k++ ))
#do /
    for j in 1 2 3 4 5
    do
	mkdir models/denver/${foldname}_adaboost${j}_set$k//
        mkdir /media/datadisk/database/Denver/data_steps/testResult/${foldname}_adaboost${j}_set${k}/
        rm /media/datadisk/database/Denver/data_steps/testResult/${foldname}_adaboost${j}_set${k}/*
        for i in  0 1 2 3 4 5 6 7 8 9 10 11
        do
        curAU=${aulist[${i}]}
        ./models/denver/solver_train_adaboosttry.sh ${foldname} ${curAU} ${j} $k
        ./build/tools/caffe train -gpu 0 -solver=models/denver/${foldname}_adaboost${j}_set${k}/solver${curAU}.prototxt  #-weights=/media/datadisk/database/FERA2015/data_steps/CNNData/normWarpImage/AU_binary_fromFera2015.caffemodel
        done
    done
#done
