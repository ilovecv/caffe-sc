#!/usr/bin/env sh

foldnum=$1
aulist=(2 12 25)
    for j in 1 2 3 4 5
    do
        foldname=${foldnum}_adaboostcross$j
        mkdir models/cross/${foldname}/
        mkdir /media/datadisk/database/Denver/data_steps/testResult/${foldname}/
        rm /media/datadisk/database/Denver/data_steps/testResult/${foldname}/*
        for i in 0 1 2
        do
        curAU=${aulist[${i}]}
        ./models/cross/solver_train_adaboost.sh ${curAU} ${foldname} 
        ./build/tools/caffe train -gpu 0 -solver=models/cross/${foldname}/solver${curAU}.prototxt
        done
    done


	#./build/tools/caffe test -gpu 0 -model models/cross/${datafile}_binarycross$j//train_val${curau}.prototxt -weights=/media/datadisk/database/cross/data_steps/CNNData/normWarpImage/AU_binaryeu_fromcross.caffemodel -iterations 145
