#!/usr/bin/env sh

foldname=normImage
set=$2
aulist=(1 2 4 5 6 7 9 12 15 17 23 24 25 27)
#testiter=(11 10 9 10 9 9 10 11)
dir=/home/handong/Shizhong/cmu/data_steps/
datadir=$dir/DRMF_CUDA/
#for (( k=1; k<3; k++ ))
#do 
    for run in 1 2 3 4 5
    do
	resdir=$dir/testResult/${foldname}_adaptive3fix444_set${set}_run${run}/
	mkdir $resdir
        rm $resdir/*
        for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13
        do
        curAU=${aulist[${i}]}
	scorefile=$resdir/outscore_baseline${curAU}.txt
        rm $scorefile
        if [ ! -e "$scorefile" ]
        then
    	    echo "$scorefile not exist"
            ./models/cmu/solver_train_adaptive.sh ${curAU} ${set} ${datadir} ${resdir} ${foldname} 11
            ./build/tools/caffe train -gpu $1 -solver=$resdir/solver${curAU}.prototxt  #-weights=/media/datadisk/database/FERA2015/data_steps/CNNData/normWarpImage/AU_binary_fromFera2015.caffemodel
        fi
        done
    done
#done
