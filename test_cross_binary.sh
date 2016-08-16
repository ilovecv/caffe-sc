#!/usr/bin/env sh

foldnum=$1
aulist=(2 12 25)
for j in 1 2 3 4 5
do
    foldname=${foldnum}_binarycross$j
    mkdir models/cross/${foldname}/
    mkdir /media/datadisk/database/Denver/data_steps/testResult/${foldname}/
    rm /media/datadisk/database/Denver/data_steps/testResult/${foldname}/outscore*
    for i in 0 1 2
    do
    curAU=${aulist[${i}]}
	for k in 1 2 3 4 5 6 7 8 9
	do
    	./models/cross/solver_test_binary.sh ${curAU} ${foldname} $k
    	./build/tools/caffe test -gpu 0 -model models/cross/${foldname}/test_set${k}_au${curAU}_binary.prototxt -weights=/media/datadisk/database/Denver/data_steps/testResult/${foldname}/train${curAU}__iter_1600.caffemodel -iterations 145
	done
    done
done
