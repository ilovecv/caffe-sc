#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  int bottom_channels_ = bottom[0]->channels();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  weight_vec_.Reshape(1,1,1,bottom_channels_);
  weight_rep_vec_.Reshape(bottom[0]->shape());
  Dtype* weight = weight_vec_.mutable_cpu_data();
  Dtype* weight_rep=weight_rep_vec_.mutable_cpu_data();
  caffe_set(bottom_channels_, Dtype(0), weight);
  const Dtype* target = bottom[1]->cpu_data();
  for (int i =0; i <num; i++){
  	for (int j=0; j<bottom_channels_; j++){
  		weight[j]+=target[i*bottom_channels_+j];
  	}
  }
    //printf("weight:");
  for(int j=0; j<bottom_channels_; j++){
  	weight[j]=sqrt(weight[j]/(num-weight[j]));
  }
  for(int j=0; j<count; j++){
  	if(target[j]==1){
  		weight_rep[j]=1;
  	}
  	else{
  		weight_rep[j]=weight[j%bottom_channels_];
  	}
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_mul(bottom[0]->count(), weight_rep_vec_.cpu_data(),  diff_.mutable_cpu_data(),  diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
      //caffe_mul(bottom[i]->count(), weight_rep_vec_.cpu_data(),  bottom[i]->mutable_cpu_diff(),  bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe
