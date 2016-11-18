#ifndef CAFFE_BASE_ADAPTIVE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_ADAPTIVE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include<list>

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseAdaptiveConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseAdaptiveConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void ReshapeFilterUpDown();
  void weights_updown_forward();
  void update_kerneldiff_quene();
  void weights_cut(int weight_channel_offset_,int kernel_int_size,Dtype *weight);
  void weights_pad(int weight_channel_offset_,int kernel_int_size,Dtype *weight);
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights_up, const Dtype* weights_down,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights_up,const Dtype* weights_down,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights_up, Dtype* weights_down);
  void backward_cpu_kernel_size(const Dtype* output_diff,const Dtype* input,
       const Dtype* weights_up,const Dtype* weights_down, Dtype* kernel_size_diff);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights_up,const Dtype* weights_down,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights_up,const Dtype* weights_down,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype* weights_up, Dtype* weights_down);
  void backward_gpu_kernel_size(const Dtype* output_diff,const Dtype* input,
		  const Dtype* weights_up,const Dtype* weights_down, Dtype* kernel_size_diff);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_up_;
  Blob<Dtype> kernel_shape_float_;
  Blob<int> kernel_shape_;
  Blob<int> kernel_shape_max_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;
  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int init_kernel_width_;
  int init_kernel_height_;
  int weight_offset_;
  int weight_channel_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;
  bool Debug_;
  int check_iteration_;
  Blob<Dtype> weightone_multiplier_;


 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        im2col_cpu(data, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_max_.cpu_data()[0], kernel_shape_max_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
      } else {
        im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
            col_buffer_shape_.data(), kernel_shape_max_.cpu_data(),
            pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
      }
    }
    inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        col2im_cpu(col_buff, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_max_.cpu_data()[0], kernel_shape_max_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
      } else {
        col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
            col_buffer_shape_.data(), kernel_shape_max_.cpu_data(),
            pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
      }
    }
  #ifndef CPU_ONLY
    inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        im2col_gpu(data, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_max_.cpu_data()[0], kernel_shape_max_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
      } else {
        im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
            conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
            kernel_shape_max_.gpu_data(), pad_.gpu_data(),
            stride_.gpu_data(), dilation_.gpu_data(), col_buff);
      }
    }
    inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        col2im_gpu(col_buff, conv_in_channels_,
            conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
            kernel_shape_max_.cpu_data()[0], kernel_shape_max_.cpu_data()[1],
            pad_.cpu_data()[0], pad_.cpu_data()[1],
            stride_.cpu_data()[0], stride_.cpu_data()[1],
            dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
      } else {
        col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
            conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
            kernel_shape_max_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
            dilation_.gpu_data(), data);
      }
    }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;
  int iter_;
  int seqnum_;
  float up_ratio_;
  float down_ratio_;


  AdaptiveConvolutionParameter conv_param_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> kernel_diff_buffer_;
  Blob<Dtype> weight_filter_up_;
  Blob<Dtype> weight_filter_down_;
  Blob<Dtype> weight_ratio_up_;
  Blob<Dtype> weight_ratio_down_;
  Blob<Dtype> weight_multiplier_;
  Blob<Dtype> output_multiplier_;
  Blob<Dtype> bias_multiplier_;
  Blob<int> kernel_taken_;
  Blob<int> fixsize_;
  std::list<Dtype> diffhistory_;



};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
