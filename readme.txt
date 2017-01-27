Add the boosting layer to caffe
1: put the boosting_layer.cpp under the src/caffe/layers
2: put the boosting_layer.hpp under the include/caffe/layers
3: put the adaBoostBinay.cpp and basefunc.cpp under the include/caffe/util
4: put the adaBoostBiany.hpp and basefunc.hpp under the include/caffe/util
5: change the caffe.proto
  optional BoostingParameter boosting_param = 146;
  message BoostingParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional float imith = 2 [default=0.1];

  // The first axis to be lumped into a single inner product computation;
  // all preceding axes are retained in the output.
  // May be negative to index from the end (e.g., -1 for the last axis).
  optional int32 axis = 5 [default = 1];
}
