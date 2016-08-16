#!/usr/bin/env sh

foldname=$1
mkdir models/fera2015/${foldname}/
mkdir /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}/
rm /media/datadisk/database/FERA2015/data_steps/testResult/${foldname}/*

for i in 1 #2 3 4 5 6 7 8 9 10
do
./models/fera2015/solver_train_multi.sh ${foldname} ${i}
#./models/fera2015/solver_train_1conv.sh ${foldname} ${i}
./build/tools/caffe train -gpu $2 -solver=models/fera2015/${foldname}/solver${i}.prototxt #-weights models/fera2015/landmarks_modify.caffemodel
done
