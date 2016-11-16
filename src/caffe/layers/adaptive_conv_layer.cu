#include <vector>

#include "caffe/layers/adaptive_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void AdaptiveConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weights_up = this->blobs_[0]->gpu_data();
  const Dtype* weights_down = this->blobs_[1]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weights_up, weights_down,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[2]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void AdaptiveConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weights_up = this->blobs_[0]->gpu_data();
  const Dtype* weights_down = this->blobs_[1]->gpu_data();
  Dtype* weight_diff_up = this->blobs_[0]->mutable_gpu_diff();
  Dtype* weight_diff_down = this->blobs_[1]->mutable_gpu_diff();
  Dtype* kernel_size_diff = this->blobs_[3]->mutable_gpu_diff();
  //caffe_gpu_set(this->kernel_shape_float_.count(),Dtype(0),kernel_size_diff);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[2]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff_up,weight_diff_down);
          this->backward_gpu_kernel_size(top_diff+n*this->top_dim_, bottom_data + n*this->bottom_dim_,
              weights_up,weights_down, kernel_size_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weights_up, weights_down,
              bottom_diff + n * this->bottom_dim_);
        }
      }////
    }
  }
  this->update_kerneldiff_quene();
}

INSTANTIATE_LAYER_GPU_FUNCS(AdaptiveConvolutionLayer);

}  // namespace caffe
