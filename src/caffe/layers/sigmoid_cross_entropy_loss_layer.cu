#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
	//printf("new sigmoid output:\n");
	//for(int j=0; j<100;j++){
	//      printf("%f ", sigmoid_output_->cpu_data()[j]);
	//}
	//printf("\n");
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    const Dtype* weight_rep=weight_rep_vec_.gpu_data();
    const Dtype* temp1 = temp1_vec_.gpu_data();
    const Dtype* temp2 = temp2_vec_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    caffe_gpu_mul(count, sigmoid_output_data, temp2, bottom_diff);
    caffe_gpu_add(count, bottom_diff, temp1, bottom_diff);
    // Scale down gradient

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);

  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
