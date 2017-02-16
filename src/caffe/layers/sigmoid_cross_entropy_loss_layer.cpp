#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  bottom_channels_ = bottom[0]->channels();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  //printf("bottom_channels=%d,count=%d,num=%d\n",bottom_channels_,count,num);
  weight_vec_.Reshape(1,1,1,bottom_channels_);
  weight_rep_vec_.Reshape(bottom[0]->shape());
  temp1_vec_.Reshape(bottom[0]->shape());
  temp2_vec_.Reshape(bottom[0]->shape());
  Dtype* weight = weight_vec_.mutable_cpu_data();
  Dtype* weight_rep=weight_rep_vec_.mutable_cpu_data();
  Dtype* temp1 = temp1_vec_.mutable_cpu_data();
  Dtype* temp2 = temp2_vec_.mutable_cpu_data();
  caffe_set(bottom_channels_, Dtype(0), weight);
  const Dtype* target = bottom[1]->cpu_data();
  for (int i =0; i <num; i++){
	  for (int j=0; j<bottom_channels_; j++){
		  weight[j]+=target[i*bottom_channels_+j];
	  }
  }
  //printf("num=%d, bottom_channels=%d, count=%d\n",num, bottom_channels, count);
  //printf("balance weight:");
  for(int j=0; j<bottom_channels_; j++){
	  weight[j]=weight[j]/num;
	  //printf("%f ", weight[j]);
  }
  //printf("\n");
  for(int j=0; j<count; j++){
	  weight_rep[j]=weight[j%bottom_channels_];
  }
  caffe_mul(count, weight_rep, target, temp1);
  caffe_add(count, weight_rep, target, temp2);
  caffe_sub(count, temp2, temp1, temp2);
  caffe_sub(count, temp2, temp1, temp2);
  caffe_sub(count, temp1, target, temp1);

  /*for(int j=0; j<100;j++){
 	  printf("%f ", temp1[j]);
   }
   printf("\n");
  for(int j=0; j<100;j++){
	  printf("%f ", temp2[j]);
  }
  printf("\n");*/

}
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  //Dtype* weight = weight_vec_.mutable_cpu_data();
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_ );
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* weight = weight_vec_.cpu_data();
  Dtype loss = 0;
  Dtype curloss=0;
  //printf("bottom_channels=%d\n",bottom_channels_);
  //printf("curloss:\n");
  for (int i = 0; i < count; ++i) {
	  //printf("%f\n",weight[i%bottom_channels_]);
	Dtype w=weight[i%bottom_channels_];
    if(input_data[i]>=0){
    	curloss=(2*w*target[i]-w-target[i])*log(1+exp(-input_data[i]))-input_data[i]*w*(1-target[i]);
    }
    else{
    	curloss=(2*w*target[i]-w-target[i])*log(1+exp(input_data[i]))+input_data[i]*target[i]*(1-w);
    }
    loss-=curloss;
	//loss-= input_data[i] * (target[i] - (input_data[i] >= 0)) -
	//    log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));


  }
  //printf("newloss=%f oldloss=%f\n", loss, oldloss);
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* temp1 = temp1_vec_.cpu_data();
    const Dtype* temp2 = temp2_vec_.cpu_data();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_mul(count, sigmoid_output_data, temp2, bottom_diff);
    caffe_add(count, bottom_diff, temp1, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
