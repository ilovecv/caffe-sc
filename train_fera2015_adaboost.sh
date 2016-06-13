#!/usr/bin/env sh

foldname=$1
mkdir models/fera2015/${foldname}_adaboost/
aulist=(2 12 17 25 28 45)
for j in 1 #2 3 4 5 6 7 8 9 10
do
    mkdir /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}_adaboost/
    rm /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}_adaboost/*
    for i in 0 1 2 3 4 5 
    do
    curAU=${aulist[${i}]}
    ./models/fera2015/solver_train_resadaboost.sh ${foldname} ${curAU}
    #./models/fera2015/solver_train_1conv.sh ${foldname} ${i}
    ./build/tools/caffe train -gpu $2 -solver=models/fera2015/${foldname}_adaboost/solver${curAU}.prototxt  #-weights=/media/datadisk/database/FERA2015/data_steps/CNNData/normWarpImage/AU_binary_fromFera2015.caffemodel
    done
    #mv /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}_adaboost /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}_adaboost${j}
done
