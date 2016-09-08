#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_adaptive_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  conv_param_ = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param_.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param_.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  kernel_shape_up_.Reshape(spatial_dim_blob_shape);
  kernel_shape_down_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param_.has_kernel_h() || conv_param_.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param_.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param_.kernel_h();
    kernel_shape_data[1] = conv_param_.kernel_w();
  } else {
    const int num_kernel_dims = conv_param_.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param_.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param_.has_stride_h() || conv_param_.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param_.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param_.stride_h();
    stride_data[1] = conv_param_.stride_w();
  } else {
    const int num_stride_dims = conv_param_.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param_.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param_.has_pad_h() || conv_param_.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param_.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param_.pad_h();
    pad_data[1] = conv_param_.pad_w();
  } else {
    const int num_pad_dims = conv_param_.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param_.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param_.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param_.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  // - blobs_[2] holds the
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this -> blobs_.resize(3);
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  vector<int> kernel_size_shape(1, 1);
  this->blobs_[2].reset(new Blob<Dtype>(kernel_size_shape));
  this->blobs_[2]->mutable_cpu_data()[0]=kernel_shape_data[0];
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);//the number of images
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  //create the weight_up and weight_down
  //in future, we can check whether it kernel size is changed
  int* kernel_shape_data_up = kernel_shape_up_.mutable_cpu_data();
  int* kernel_shape_data_down = kernel_shape_down_.mutable_cpu_data();
  float kernel_size = this->blobs_[2]->cpu_data()[0];
    if (conv_param_.has_kernel_h() || conv_param_.has_kernel_w()) {
      CHECK_EQ(num_spatial_axes_, 2)
          << "kernel_h & kernel_w can only be used for 2D convolution.";
      CHECK_EQ(0, conv_param_.kernel_size_size())
          << "Either kernel_size or kernel_h/w should be specified; not both.";
      kernel_shape_data_up[0] = round(kernel_size)+2;
      kernel_shape_data_up[1] = round(kernel_size)+2;
      kernel_shape_data_down[0] = round(kernel_size)-2;
      kernel_shape_data_down[1] = round(kernel_size)-2;
    } else {
      const int num_kernel_dims = conv_param_.kernel_size_size();
      CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
          << "kernel_size must be specified once, or once per spatial dimension "
          << "(kernel_size specified " << num_kernel_dims << " times; "
          << num_spatial_axes_ << " spatial dims).";
        for (int i = 0; i < num_spatial_axes_; ++i) {
          kernel_shape_data_up[i] =
              round(kernel_size)+2;
          kernel_shape_data_down[i] =
              round(kernel_size)-2;
        }
    }
    for (int i = 0; i < num_spatial_axes_; ++i) {
      CHECK_GT(kernel_shape_data_up[i], 0) << "Filter dimensions must be nonzero.";
      CHECK_GT(kernel_shape_data_down[i], 0) << "Filter dimensions must be nonzero.";
  }
  vector<int> weight_shape_up(2);
  vector<int> weight_shape_down(2);
  weight_shape_up[0] = conv_out_channels_;
  weight_shape_up[1] = conv_in_channels_ / group_;
  weight_shape_down[0] = conv_out_channels_;
  weight_shape_down[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape_up.push_back(kernel_shape_data_up[i]);
    weight_shape_down.push_back(kernel_shape_data_down[i]);
  }
  weight_filter_up_.Reshape(weight_shape_up);
  weight_filter_down_.Reshape(weight_shape_down);
  kernel_dim_up_ = weight_filter_up_.count(1);//conv_in_channels*k_h*k_w
  kernel_dim_down_ =weight_filter_down_.count(1);
  weight_offset_up_ = conv_out_channels_ * kernel_dim_up_ / group_;//conv_out_channels*conv_in_channels*k_h*k_w
  weight_offset_down_ = conv_out_channels_ * kernel_dim_down_ / group_;
  //Pad and cut the weight of up and down
  weights_pad_forward();
  weights_cut_forward();
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_up_ = kernel_dim_up_ * conv_out_spatial_dim_;
  col_offset_down_ = kernel_dim_down_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_up_.clear();
  col_buffer_shape_up_.push_back(kernel_dim_up_ * group_);
  col_buffer_shape_down_.clear();
  col_buffer_shape_down_.push_back(kernel_dim_up_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_up_.push_back(input_shape(i + 1));
      col_buffer_shape_down_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_up_.push_back(output_shape_[i]);
      col_buffer_shape_down_.push_back(output_shape_[i]);
    }
  }
  col_buffer_up_.Reshape(col_buffer_shape_up_);
  col_buffer_down_.Reshape(col_buffer_shape_down_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weights_pad_forward(){
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_up = weight_filter_up_.mutable_cpu_data();
  CHECK_EQ(this->blobs_[0]->shape(0), weight_filter_up_.shape(0)) << "The out put filter number should be same.";
  CHECK_EQ(this->blobs_[0]->shape(1), weight_filter_up_.shape(1)) << "The input filter number should be same.";
  int ker_num = this->blobs_[0] -> count(0,2);
  int ker_height = this->blobs_[0]->height();
  int ker_height_up = weight_filter_up_.height();
  int ker_width = this->blobs_[0]->width();
  int ker_width_up = weight_filter_up_.width();
  int width_start=(ker_width_up-ker_width)/2;
  int height_start=(ker_height_up-ker_height)/2;
  CHECK_GE(width_start, 0) << "The width of up filter should bigger than the original filter";
  CHECK_GE(height_start, 0) << "The height of up filter should bigger than the original filter";
  CHECK_LE(width_start, 1) << "The width of up filter should bigger than the original filter within 2 pixel";
  CHECK_LE(height_start, 1) << "The height of up filter should bigger than the original filter within 2 pixel";
  int ker_offset = ker_width*ker_height;
  int ker_offset_up= ker_width_up*ker_height_up;
  for (int i=0; i< ker_num; i++){
    for(int j=0; j< ker_height; j++){
      for(int k=0; k< ker_width; k++){
        weight_up[i*ker_offset_up+(j+height_start)*ker_width_up+k+width_start]=weight[i*ker_offset+j*ker_width+k];
      }
      if(width_start==1){
        weight_up[i*ker_offset_up+(j+height_start)*ker_width_up]=weight[i*ker_offset+j*ker_width];//first column
        weight_up[i*ker_offset_up+(j+height_start)*ker_width_up+ker_width_up-1]=weight[i*ker_offset+j*ker_width+ker_width-1];//last column
      }
    }
    if(height_start==1){
      for(int k=0; k< ker_width_up;k++ ){
        weight_up[i*ker_offset_up+k]=weight_up[i*ker_offset_up+ker_width_up+k];//first row
        weight_up[i*ker_offset_up+(ker_height_up-1)*ker_width_up+k]=weight_up[i*ker_offset_up+(ker_height_up-2)*ker_width_up+k];//last row
      }
    }
  }
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weights_pad_backward(){
  Dtype* weight = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* weight_down = weight_filter_down_.cpu_diff();
  CHECK_EQ(this->blobs_[0]->shape(0), weight_filter_down_.shape(0)) << "The out put filter number should be same.";
  CHECK_EQ(this->blobs_[0]->shape(1), weight_filter_down_.shape(1)) << "The input filter number should be same.";
  int ker_num = this->blobs_[0] -> count(0,2);
  int ker_height = this->blobs_[0]->height();
  int ker_height_down = weight_filter_down_.height();
  int ker_width = this->blobs_[0]->width();
  int ker_width_down = weight_filter_down_.width();
  int width_start=(ker_width-ker_width_down)/2;
  int height_start=(ker_height-ker_height_down)/2;
  CHECK_GE(width_start, 0) << "The width of down filter should smaller than the original filter";
  CHECK_GE(height_start, 0) << "The height of down filter should smaller than the original filter";
  CHECK_LE(width_start, 1) << "The width of down filter should smaller than the original filter within 2 pixel";
  CHECK_LE(height_start, 1) << "The height of down filter should smaller than the original filter within 2 pixel";
  int ker_offset = ker_width*ker_height;
  int ker_offset_down= ker_width_down*ker_height_down;
  for (int i=0; i< ker_num; i++){
    for(int j=0; j< ker_height_down; j++){
      for(int k=0; k< ker_width_down; k++){
        weight[i*ker_offset+(j+height_start)*ker_width+k+width_start]=weight_down[i*ker_offset_down+j*ker_width_down+k];
      }
      if(width_start==1){
        weight[i*ker_offset+(j+height_start)*ker_width]=weight_down[i*ker_offset_down+j*ker_width_down];//first column
        weight[i*ker_offset+(j+height_start)*ker_width+ker_width-1]=weight_down[i*ker_offset_down+j*ker_width_down+ker_width_down-1];//last column
      }
    }
    if(height_start==1){
      for(int k=0; k< ker_width;k++ ){
        weight[i*ker_offset+k]=weight[i*ker_offset+ker_width+k];//first row
        weight[i*ker_offset+(ker_height-1)*ker_width+k]=weight[i*ker_offset+(ker_height-2)*ker_width+k];//last row
      }
    }
  }
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weights_cut_forward(){
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_down = weight_filter_down_.mutable_cpu_data();
  CHECK_EQ(this->blobs_[0]->shape(0), weight_filter_down_.shape(0)) << "The out put filter number should be same.";
  CHECK_EQ(this->blobs_[0]->shape(1), weight_filter_down_.shape(1)) << "The input filter number should be same.";
  int ker_num = this->blobs_[0] -> count(0,2);
  int ker_height = this->blobs_[0]->height();
  int ker_height_down = weight_filter_down_.height();
  int ker_width = this->blobs_[0]->width();
  int ker_width_down = weight_filter_down_.width();
  int width_start=(ker_width-ker_width_down)/2;
  int height_start=(ker_height-ker_height_down)/2;
  CHECK_GE(width_start, 0) << "The width of down filter should smaller than the original filter";
  CHECK_GE(height_start, 0) << "The height of down filter should smaller than the original filter";
  CHECK_LE(width_start, 1) << "The width of down filter should smaller than the original filter within 2 pixel";
  CHECK_LE(height_start, 1) << "The height of down filter should smaller than the original filter within 2 pixel";
  int ker_offset = ker_width*ker_height;
  int ker_offset_down= ker_width_down*ker_height_down;
  for (int i=0; i< ker_num; i++){
    for(int j=0; j< ker_height_down; j++){
      for(int k=0; k< ker_width_down; k++){
        weight_down[i*ker_offset_down+j*ker_width_down+k]=weight[i*ker_offset+(j+height_start)*ker_width+k+width_start];
      }
    }
  }
}


template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weights_cut_backward(){
  Dtype* weight = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* weight_up = weight_filter_up_.cpu_diff();
  CHECK_EQ(this->blobs_[0]->shape(0), weight_filter_up_.shape(0)) << "The out put filter number should be same.";
  CHECK_EQ(this->blobs_[0]->shape(1), weight_filter_up_.shape(1)) << "The input filter number should be same.";
  int ker_num = this->blobs_[0] -> count(0,2);
  int ker_height = this->blobs_[0]->height();
  int ker_height_up = weight_filter_up_.height();
  int ker_width = this->blobs_[0]->width();
  int ker_width_up = weight_filter_up_.width();
  int width_start=(ker_width_up-ker_width)/2;
  int height_start=(ker_height_up - ker_height)/2;
  CHECK_GE(width_start, 0) << "The width of up filter should bigger than the original filter";
  CHECK_GE(height_start, 0) << "The height of up filter should bigger than the original filter";
  CHECK_LE(width_start, 1) << "The width of up filter should bigger than the original filter within 2 pixel";
  CHECK_LE(height_start, 1) << "The height of up filter should bigger than the original filter within 2 pixel";
  int ker_offset = ker_width*ker_height;
  int ker_offset_up= ker_width_up*ker_height_up;
  for (int i=0; i< ker_num; i++){
    for(int j=0; j< ker_height; j++){
      for(int k=0; k< ker_width; k++){
        weight[i*ker_offset+j*ker_width+k]=weight_up[i*ker_offset_up+(j+height_start)*ker_width_up+k+width_start];
      }
    }
  }
}


template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff_up = input;
  const Dtype* col_buff_down = input;
  const Dtype* weights_up = weight_filter_up_.cpu_data();
  const Dtype* weights_down = weight_filter_down_.cpu_data();
  //add another caffe_cpu_gemm with the
  float kernel_size = this->blobs_[2]->cpu_data()[0];
  float up_ratio = ceil(kernel_size)-kernel_size;
  float down_ratio = kernel_size-floor(kernel_size);
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_up_cpu(input, col_buffer_up_.mutable_cpu_data());
      conv_im2col_down_cpu(input, col_buffer_down_.mutable_cpu_data());
    }
    col_buff_up = col_buffer_up_.cpu_data();
    col_buff_down = col_buffer_down_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_up_,
        (Dtype)1., weights_up + weight_offset_up_ * g, col_buff_up + col_offset_up_ * g,
        (Dtype)0., output + output_offset_ * g);//with kernal size round(k)+2
    //the kernel_dim_, weight_offset_, col_offset_ will be different
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_down_,
        (Dtype)up_ratio, weights_down + weight_offset_down_ * g, col_buff_down + col_offset_down_ * g,
        (Dtype)down_ratio, output + output_offset_ * g);//with kernal size round(k)-2
  }
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff_up = col_buffer_up_.mutable_cpu_data();
  Dtype* col_buff_down = col_buffer_down_.mutable_cpu_data();
  const Dtype* weights_up = weight_filter_up_.cpu_data();
  const Dtype* weights_down = weight_filter_down_.cpu_data();
  float kernel_size = this->blobs_[2]->cpu_data()[0];
  float up_ratio = ceil(kernel_size)-kernel_size;
  float down_ratio = kernel_size-floor(kernel_size);
  if (is_1x1_) {
    col_buff_up = input;
    col_buff_down = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_up_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights_up + weight_offset_up_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff_up + col_offset_up_ * g);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_down_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights_down + weight_offset_down_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff_down + col_offset_down_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_up_cpu(col_buff_up,0.0,up_ratio, input);
    conv_col2im_down_cpu(col_buff_down,1.0, down_ratio, input);
  }
}
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weight_add_updown(const Dtype* weight_diff_up, float ratio_up,
    const Dtype* weight_diff_down, float ratio_down, int length, Dtype* weight_diff) {
	int ker_height = this->blobs_[0]->height();
	int ker_width = this->blobs_[0]->width();
	int ker_offset = ker_width*ker_height;
	int ker_height_up = weight_filter_up_.height();
	int ker_width_up = weight_filter_up_.width();
    int ker_offset_up = ker_width_up * ker_height_up;
    int ker_height_down = weight_filter_down_.height();
    int ker_width_down = weight_filter_down_.width();
    int ker_offset_down = ker_width_down * ker_height_down;
    int width_start_up=(ker_width_up-ker_width)/2;
    int height_start_up=(ker_height_up - ker_height)/2;
    int width_start_down=(ker_width-ker_width_down)/2;
    int height_start_down=(ker_height-ker_height_down)/2;
	int ker_num = length/ker_offset;
	for (int i=0; i< ker_num; i++){
	    for(int j=0; j< ker_height_down; j++){
	      for(int k=0; k< ker_width_down; k++){
	        weight_diff[i*ker_offset+(j+height_start_down)*ker_width+k+width_start_down]=ratio_down*weight_diff_down[i*ker_offset_down+j*ker_width_down+k];
	      }
	      if(width_start_down==1){
	        weight_diff[i*ker_offset+(j+height_start_down)*ker_width]=ratio_down*weight_diff_down[i*ker_offset_down+j*ker_width_down];//first column
	        weight_diff[i*ker_offset+(j+height_start_down)*ker_width+ker_width-1]=ratio_down*weight_diff_down[i*ker_offset_down+j*ker_width_down+ker_width_down-1];//last column
	      }
	    }
	    if(height_start_down==1){
	      for(int k=0; k< ker_width;k++ ){
	        weight_diff[i*ker_offset+k]=weight_diff[i*ker_offset+ker_width+k];//first row
	        weight_diff[i*ker_offset+(ker_height-1)*ker_width+k]=weight_diff[i*ker_offset+(ker_height-2)*ker_width+k];//last row
	      }
	    }
	}
	for (int i=0; i< ker_num; i++){
	    for(int j=0; j< ker_height; j++){
	      for(int k=0; k< ker_width; k++){
	        weight_diff[i*ker_offset+j*ker_width+k] += ratio_up * weight_diff_up[i*ker_offset_up+(j+height_start_up)*ker_width_up+k+width_start_up];
	      }
	    }
	}
}



template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff_up = input;
  const Dtype* col_buff_down = input;
  Dtype* weights_diff_up = weight_filter_up_.mutable_cpu_diff();
  Dtype* weights_diff_down = weight_filter_down_.mutable_cpu_diff();
  float kernel_size = this->blobs_[2]->cpu_data()[0];
  float up_ratio = ceil(kernel_size)-kernel_size;
  float down_ratio = kernel_size-floor(kernel_size);
  if (!is_1x1_) {
    conv_im2col_up_cpu(input, col_buffer_up_.mutable_cpu_data());
    col_buff_up = col_buffer_up_.cpu_data();
    conv_im2col_down_cpu(input, col_buffer_down_.mutable_cpu_data());
    col_buff_down = col_buffer_down_.cpu_data();

  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_up_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff_up + col_offset_up_ * g,
        (Dtype)1., weights_diff_up + weight_offset_up_ * g);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_down_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff_down + col_offset_down_ * g,
        (Dtype)1., weights_diff_down + weight_offset_down_ * g);
    weight_add_updown(weights_diff_up + weight_offset_up_ * g, up_ratio, weights_diff_down +
    	weight_offset_down_ * g, down_ratio, weight_offset_ * g,weights + weight_offset_ * g);
  }

}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_cpu_kernel_size(const Dtype* output_diff,const Dtype* input,
	    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff_up = input;
  const Dtype* col_buff_down = input;
	  //add another caffe_cpu_gemm with the
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_up_cpu(input, col_buffer_up_.mutable_cpu_data());
      conv_im2col_down_cpu(input, col_buffer_down_.mutable_cpu_data());
    }
    col_buff_up = col_buffer_up_.cpu_data();
    col_buff_down = col_buffer_down_.cpu_data();
  }
  Dtype sum=0.0;
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_up_,
        (Dtype)1., weights + weight_offset_up_ * g, col_buff_up + col_offset_up_ * g,
        (Dtype)0., output + output_offset_ * g);//with kernal size round(k)+2
    //the kernel_dim_, weight_offset_, col_offset_ will be different
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_down_,
        (Dtype)0.25, weights + weight_offset_down_ * g, col_buff_down + col_offset_down_ * g,
        (Dtype)(-0.25), output + output_offset_ * g);//with kernal size round(k)-2
    caffe_mul<Dtype>(output_offset_ * g, output_diff+ output_offset_ * g,
    	output+ output_offset_ * g, output+ output_offset_ * g);
    for(int i=0;i<output_offset_;i++){
       sum+=output[output_offset_ * g + i];
    }
  }
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

//template <typename Dtype>
//void BaseAdaptiveConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
//    const Dtype* weights, Dtype* output, bool skip_im2col) {
//  const Dtype* col_buff = input;
//  if (!is_1x1_) {
//    if (!skip_im2col) {
//      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
//    }
//    col_buff = col_buffer_.gpu_data();
//  }
//  for (int g = 0; g < group_; ++g) {
//    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
//        group_, conv_out_spatial_dim_, kernel_dim_up_,
//        (Dtype)1., weights + weight_offset_up_ * g, col_buff + col_offset_down_ * g,
//        (Dtype)0., output + output_offset_ * g);
//    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
//            group_, conv_out_spatial_dim_, kernel_dim_down_,
//            (Dtype)1., weights + weight_offset_down_ * g, col_buff + col_offset_down_ * g,
//            (Dtype)0., output + output_offset_ * g);
//  }
//}
//
//template <typename Dtype>
//void BaseAdaptiveConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
//    const Dtype* bias) {
//  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
//      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
//      (Dtype)1., output);
//}
//
//template <typename Dtype>
//void BaseAdaptiveConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
//    const Dtype* weights, Dtype* input) {
//  Dtype* col_buff = col_buffer_.mutable_gpu_data();
//  if (is_1x1_) {
//    col_buff = input;
//  }
//  for (int g = 0; g < group_; ++g) {
//    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
//        conv_out_spatial_dim_, conv_out_channels_ / group_,
//        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
//        (Dtype)0., col_buff + col_offset_ * g);
//  }
//  if (!is_1x1_) {
//    conv_col2im_gpu(col_buff, input);
//  }
//}
//
//template <typename Dtype>
//void BaseAdaptiveConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
//    const Dtype* output, Dtype* weights) {
//  const Dtype* col_buff = input;
//  if (!is_1x1_) {
//    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
//    col_buff = col_buffer_.gpu_data();
//  }
//  for (int g = 0; g < group_; ++g) {
//    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
//        kernel_dim_, conv_out_spatial_dim_,
//        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
//        (Dtype)1., weights + weight_offset_ * g);
//  }
//}
//
//template <typename Dtype>
//void BaseAdaptiveConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
//    const Dtype* input) {
//  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
//      input, bias_multiplier_.gpu_data(), 1., bias);
//}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseAdaptiveConvolutionLayer);

}  // namespace caffe
