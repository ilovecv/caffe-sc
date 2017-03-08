#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_adaptive_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;

namespace caffe {

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  conv_param_ = this->layer_param_.adaptiveconvolution_param();
  force_nd_im2col_ = conv_param_.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param_.axis());
  num_output_ = conv_param_.num_output();
  CHECK_GT(num_output_, 0);
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);//2
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  vector<int> kernel_channel_shape(1,num_output_);
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(kernel_channel_shape);
  kernel_shape_max_.Reshape(spatial_dim_blob_shape);
  kernel_shape_up_.Reshape(kernel_channel_shape);
  kernel_shape_float_.Reshape(kernel_channel_shape);
  int *kernel_shape_max_data=kernel_shape_max_.mutable_cpu_data();
  if(conv_param_.has_max_kernel_size()){
	  kernel_shape_max_data[0] = conv_param_.max_kernel_size();
	  kernel_shape_max_data[1] = conv_param_.max_kernel_size();
  }
  Dtype* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if(conv_param_.has_kernel_h() || conv_param_.has_kernel_w()) {
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
  init_kernel_width_ = kernel_shape_data[0];
  init_kernel_height_= kernel_shape_data[0];
  caffe_set(num_output_,kernel_shape_data[0],kernel_shape_data);
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
  group_ = this->layer_param_.adaptiveconvolution_param().group();
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
    weight_shape.push_back(kernel_shape_max_data[i]);//conv_out_channels_,conv_in_channels_,kernel_height_, kernel_width_
  }
  //for(int i=0; i<4; i++) printf("%d ",weight_shape[i]);printf("\n");
  bias_term_ = this->layer_param_.adaptiveconvolution_param().bias_term();
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
    if (bias_term_ && bias_shape != this->blobs_[2]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[2]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this -> blobs_.resize(5);
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.adaptiveconvolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.adaptiveconvolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }
  this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
  caffe_copy(this->blobs_[0]->count(),this->blobs_[0]->cpu_data(),this->blobs_[1]->mutable_cpu_data());
  //weight_filter_up_.Reshape(weight_shape);
  //weight_filter_down_.Reshape(weight_shape);
  //weight_ratio_up_.Reshape(weight_shape);
  //weight_ratio_down_.Reshape(weight_shape);
  //weight_multiplier_.Reshape(weight_shape);
  this->blobs_[3].reset(new Blob<Dtype>(kernel_channel_shape));
  this->blobs_[4].reset(new Blob<Dtype>(kernel_channel_shape));
  caffe_set(num_output_,(Dtype)(kernel_shape_data[0]),this->blobs_[3]->mutable_cpu_data());

  //caffe_set(num_output_,(Dtype)(kernel_shape_data[0]),this->blobs_[4]->mutable_cpu_data());
  //this->blobs_[3]->mutable_cpu_data()[0]=kernel_shape_data[0];
  kernel_dim_ = this->blobs_[0]->count(1);//conv_in_channels_,kernel_height_, kernel_width_
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  weight_channel_offset_=kernel_dim_/group_;
  vector<int> weightone_multiplier_shape(1, weight_offset_);
  weightone_multiplier_.Reshape(weightone_multiplier_shape);
  caffe_set(weightone_multiplier_.count(), Dtype(1),
          weightone_multiplier_.mutable_cpu_data());
  //printf("kernel_dim=%d,weight_offset=%d \n",kernel_dim_,weight_offset_);
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // TODO: initialize up and down size
  int* kernel_up_size=kernel_shape_up_.mutable_cpu_data();
  Dtype* kernel_down_size=this->blobs_[4]->mutable_cpu_data();
  Dtype* weights_down = this->blobs_[1]->mutable_cpu_data();
  kernel_taken_.Reshape(1,1,kernel_shape_max_data[0]+1,num_output_);
  fixsize_.Reshape(kernel_channel_shape);
  caffe_set(kernel_taken_.count(),0,kernel_taken_.mutable_cpu_data());
  caffe_set(fixsize_.count(),int(0),fixsize_.mutable_cpu_data());
  for( int t=0; t<num_output_; t++){
	  kernel_taken_.mutable_cpu_data()[(int)(kernel_down_size[t])*num_output_+t]=1;
	  kernel_down_size[t]=floor((kernel_shape_data[0]+1)/2)*2-1;
  	  kernel_up_size[t] =kernel_down_size[t]+2;
  	  //weights_cut(weight_channel_offset_,kernel_down_size[t],weights_down+weight_channel_offset_*t);
  }
  iter_=0;
  //down_ratio_=0.5;
  //up_ratio_=0.5;
  min_iter_=50;
  iter_afterflip_=0;
  Debug_ = false;

}
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weights_updown_forward(){
	  const Dtype *kernel_float_size = this->blobs_[3]->cpu_data();
	  int kernel_max_size = kernel_shape_max_.cpu_data()[0];
	  Dtype* weights_up = this->blobs_[0]->mutable_cpu_data();
	  Dtype* weights_down = this->blobs_[1]->mutable_cpu_data();
	  //const Dtype* kernel_int_size = kernel_shape_.cpu_data();
	  int* kernel_up_size=kernel_shape_up_.mutable_cpu_data();
	  const Dtype* kernel_down_size=this->blobs_[4]->cpu_data();
	  int kernel_offset = kernel_max_size*kernel_max_size;
	  int count=this->blobs_[0]->count();
	  int kernel_num = weight_channel_offset_/kernel_offset;
	  //if(Debug_){
		  //printf("weights_up:\n");
	  //for(int i=0; i<num_output_; i++){
		  //printf("%d:%f ",i+1,caffe_cpu_dot(weight_channel_offset_,weightone_multiplier_.cpu_data(),weights_up+i*weight_channel_offset_));
	  //}
	  //printf("\n");}
	  //for(int j=0; j< 9; j++){for(int k=0; k< 9; k++){ printf("%.4e ",weight[j*9+k]);} printf("\n");}
	  //caffe_copy(count,weight,weights_up);
	  //caffe_copy(count,weight,weights_down);
	  for(int t=0; t<num_output_; t++){
		  if(Debug_){
		  printf("|%d %f %f ",
		  				  kernel_up_size[t],kernel_down_size[t],kernel_float_size[t]);
		  }
		  //printf("%d%d ",kernel_up_size[t],kernel_down_size[t]);
		  //CHECK_GE(kernel_int_size[t],1)<<"the kernel int size should be bigger than zero";
		  CHECK_GE(kernel_up_size[t],1)<<"the kernel up size  should be bigger than zero";
		  CHECK_GE(kernel_down_size[t],1)<<"the kernel down size should be bigger than zero";
		  //CHECK_LE(kernel_int_size[t],kernel_max_size)<<"the kernel int size should be smaller or equal than max size";
		  CHECK_LE(kernel_up_size[t],kernel_max_size)<<"the kernel up size  should be smaller or equal than max size";
		  CHECK_LE(kernel_down_size[t],kernel_max_size)<<"the kernel down size should be smaller or equal than max size";
		  //CHECK_GE(kernel_up_size[t],kernel_int_size[t])<<"kernel up size should be bigger than the kernel int size";
		  //CHECK_LE(kernel_down_size[t],kernel_int_size[t])<<"kernel down size should be smaller or equal than the kernel int size";
		  //CHECK_GE(kernel_up_size[t],kernel_float_size[t])<<"kernel up size should be bigger than the kernel float size";
		  //CHECK_LE(kernel_down_size[t],kernel_float_size[t])<<"kernel down size should be smaller or equal than the kernel float size";

		  int kernel_shift_up=(kernel_max_size-kernel_up_size[t])/2;
		  int kernel_shift_down=(kernel_max_size-kernel_down_size[t])/2;
		  CHECK_GE(kernel_shift_up,0)<<"the kernel shift should be nonzero";
		  CHECK_GE(kernel_shift_down,0)<<"the kernel shift should be nonzero";
		  //float down_ratio = (float)(kernel_up_size[t]-kernel_float_size[t])*init_kernel_width_*
		  //		init_kernel_height_/(2*kernel_down_size[t]*kernel_down_size[t]);
		  //float up_ratio = (float)(kernel_float_size[t]-kernel_down_size[t])*init_kernel_width_*
		  //		init_kernel_height_/(2*kernel_up_size[t]*kernel_up_size[t]);
		  //float down_ratio = 0.5;//(float)(kernel_up_size[t]-kernel_float_size[t])/2;
		  //float up_ratio = 0.5;//(float)(kernel_float_size[t]-kernel_down_size[t])/2;
		  //printf("|%f %f| ",up_ratio,down_ratio);
		  for (int i=0; i< kernel_num; i++){
		  	  for(int j=0; j< kernel_max_size; j++){
		  	    for(int k=0; k< kernel_max_size; k++){
		  	  	  if(j<=kernel_shift_up-1||j>=kernel_max_size-kernel_shift_up||k<=kernel_shift_up-1||k>=kernel_max_size-kernel_shift_up){
		  	  		  weights_up[t*weight_channel_offset_+i*kernel_offset+j*kernel_max_size+k]=0;
		  	   	  }
		  	   	  if(j<=kernel_shift_down-1||j>=kernel_max_size-kernel_shift_down||k<=kernel_shift_down-1||k>=kernel_max_size-kernel_shift_down){
		  	   	  	  weights_down[t*weight_channel_offset_+i*kernel_offset+j*kernel_max_size+k]=0;
		  	   	  }
		  	    }
		  	 }
		}
	  }
	  if(Debug_){
	  printf("weight up:\n");
	  for(int j=0; j< 9; j++){for(int k=0; k< 9; k++){ printf("%.4e ",weights_up[j*9+k]);} printf("\n");}
	  printf("weight down:\n");
	  for(int j=0; j< 9; j++){for(int k=0; k< 9; k++){ printf("%.4e ",weights_down[j*9+k]);} printf("\n");}

		  //printf("\n");
		  printf("weight up sum:\n");
		  printf("%f\n",caffe_cpu_dot(count,weights_up,weightone_multiplier_.cpu_data()));
		  printf("weight down sum:\n");
		  printf("%f\n",caffe_cpu_dot(count,weights_down,weightone_multiplier_.cpu_data()));
		  printf("weight up:\n");
		  for(int i=0; i<num_output_; i++){
		  	  printf("%d:%f ",i+1,caffe_cpu_dot(weight_channel_offset_,weightone_multiplier_.cpu_data(),weights_up+i*weight_channel_offset_));
		  }
		  printf("\n");
		  printf("weight down:\n");
		  for(int i=0; i<num_output_; i++){
		  	  printf("%d:%f ",i+1,caffe_cpu_dot(weight_channel_offset_,weightone_multiplier_.cpu_data(),weights_down+i*weight_channel_offset_));
		  }
	  }
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
  if(Debug_){
  	printf("\nNew iteration %d------------------------------------------------------\n",iter_+1);
  }
  if(Debug_){
	  	int* fixsize_channel = fixsize_.mutable_cpu_data();
	    int* kernel_size_taken_=kernel_taken_.mutable_cpu_data();
   		printf("fixsize\n");
   		for(int t=0; t<num_output_;t++){
   		printf("%d ",fixsize_channel[t]);
   		}
   		printf("\n");

   		printf("size label\n");
   		for(int t=1; t<10;t++){
   		printf("%d ",kernel_size_taken_[t*num_output_]);
   		}
   		printf("\n");
  }
  // TODO: calculate the up and down kernel size
  int kernel_max_size = kernel_shape_max_.cpu_data()[0];
  if(this->phase_==TRAIN){
	  Dtype* kernel_down_size = this->blobs_[4]->mutable_cpu_data();
	  int* kernel_up_size=kernel_shape_up_.mutable_cpu_data();
	  int* kernel_size_taken_=kernel_taken_.mutable_cpu_data();
	  int* fixsize_channel = fixsize_.mutable_cpu_data();
	  Dtype* kernel_float_size=this->blobs_[3]->mutable_cpu_data();
	  Dtype *weights_up =this->blobs_[0]->mutable_cpu_data();
	  Dtype *weights_down =this->blobs_[1]->mutable_cpu_data();
	  bool sizechange=false;
	  weights_updown_forward();
	  for (int i=0; i<num_output_; i++){
		  //if(iter_<=min_iter_){

		  //}
		  if(kernel_float_size[i]>kernel_max_size) {
			  kernel_float_size[i]=kernel_max_size;
			  fixsize_channel[i]=1;
			  sizechange=true;
		  }
		  if(kernel_float_size[i]<1){
			  kernel_float_size[i]=1;
			  fixsize_channel[i]=1;
			  sizechange=true;
		  }
		  if(kernel_float_size[i]-kernel_up_size[i]>=2){
		 	 kernel_float_size[i]=kernel_up_size[i]+0.1;
		  }
		  else if(kernel_down_size[i]-kernel_float_size[i]>=2){
		     kernel_float_size[i]=kernel_down_size[i]-0.1;
		  }
		  if(kernel_float_size[i]>=kernel_up_size[i]&&kernel_up_size[i]<kernel_max_size&&fixsize_channel[i]==0){
			  caffe_copy(weight_channel_offset_,weights_up+weight_channel_offset_*i, weights_down+weight_channel_offset_*i);
			  weights_pad(weight_channel_offset_,kernel_up_size[i],weights_up+weight_channel_offset_*i);
			  kernel_down_size[i]=kernel_up_size[i];
			  kernel_up_size[i]=kernel_up_size[i]+2;
			  kernel_float_size[i]=kernel_down_size[i]+1;
			  sizechange=true;
			  //printf("hahahaha\n");
		  }
		  else if(kernel_float_size[i]<kernel_down_size[i]&&kernel_down_size[i]>1&&fixsize_channel[i]==0){
			  caffe_copy(weight_channel_offset_,weights_down+weight_channel_offset_*i, weights_up+weight_channel_offset_*i);
			  weights_cut(weight_channel_offset_,kernel_down_size[i]-2,weights_down+weight_channel_offset_*i);//3->1
			  kernel_up_size[i]=kernel_down_size[i];//3
			  kernel_down_size[i]=kernel_down_size[i]-2;//1
			  kernel_float_size[i]=kernel_down_size[i]+1;
			  sizechange=true;
			  //printf("hahahaha\n");g
		  }
		  if(sizechange==true&&fixsize_channel[i]==0){
			  kernel_size_taken_[num_output_*(int)(kernel_down_size[i])+i]+=1;
			  if(kernel_size_taken_[num_output_*(int)(kernel_down_size[i])+i]>2){
				  //fixsize_channel[i]=1;
			  }
			  iter_afterflip_=0;
		  }
		  //weights_pad(weight_channel_offset_,kernel_down_size[i],weights_up+weight_channel_offset_*i);
		  up_ratio_=(kernel_float_size[i]-kernel_down_size[i])/2;
		  down_ratio_=(kernel_up_size[i]-kernel_float_size[i])/2;
	  }
  }
  else if(this->phase_==TEST){
	  int* kernel_up_size=kernel_shape_up_.mutable_cpu_data();
	  const Dtype* kernel_down_size = this->blobs_[4]->cpu_data();
	  Dtype* kernel_float_size=this->blobs_[3]->mutable_cpu_data();
	  for (int i=0; i<num_output_; i++){
		  kernel_up_size[i]=(int)(kernel_down_size[i])+2;
		  up_ratio_=(kernel_float_size[i]-kernel_down_size[i])/2;
		  down_ratio_=(kernel_up_size[i]-kernel_float_size[i])/2;
	  }
	  weights_updown_forward();
  }
  if(Debug_){
  	  printf("up_ratio:%f, down_ratio:%f\n",up_ratio_,down_ratio_);
  }
  // TODO: to generate the weight up and weight down

  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  vector<int> kernel_diff_buffer_shape(0);
  top_shape.push_back(num_output_);
  kernel_diff_buffer_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
    kernel_diff_buffer_shape.push_back(output_shape_[i]);
  }
  kernel_diff_buffer_.Reshape(kernel_diff_buffer_shape);
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  //printf("top_shape:"); for(int i=0; i<4; i++) printf("%d ", top_shape[i]); printf("\n");
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_* conv_out_spatial_dim_;
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
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);//conv_in_channels_*kernel_height_*kernel_width_
  for (int i = 0; i < num_spatial_axes_; ++i) {
  if (reverse_dimensions()) {
        col_buffer_shape_.push_back(input_shape(i + 1));//image_height
      } else {
        col_buffer_shape_.push_back(output_shape_[i]);//image_width
      }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  //printf("onenumber=%d,outputoffest=%d\n",out_spatial_dim_,output_offset_);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
  vector<int> output_multiplier_shape(1, output_offset_);
  output_multiplier_.Reshape(output_multiplier_shape);
  caffe_set(output_multiplier_.count(), Dtype(1.0),
		  output_multiplier_.mutable_cpu_data());
//  if(Debug_){
//	std::cout<<"\n";
//	std::cout<<"diff history recent 10 iteration\n";
//    typename std::list<Dtype>::iterator it;
//    for (it=diffhistory_.begin(); it!=diffhistory_.end(); std::advance(it,num_output_)) std::cout << ' ' << *it;
//    std::cout<<"\n";
//  }
  iter_++;
  iter_afterflip_++;
}
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::update_kerneldiff_quene(){
  Dtype* kernel_float_diff=this->blobs_[3]->mutable_cpu_diff();
  //Dtype* kernel_float_size = this->blobs_[3]->mutable_cpu_data();
  int* fixsize_channel = fixsize_.mutable_cpu_data();
  for(int i=0; i<num_output_;i++){
  	if(fixsize_channel[i]==1||iter_<min_iter_)
  		kernel_float_diff[i]=0;
  	//else if(kernel_float_diff[i]>max_thresh_)
  	//	kernel_float_diff[i]=max_thresh_;
  }
  //calculate the average
  if(Debug_){
	  printf("\n");
	  printf("Kernel difference:");
	  for(int i=0; i<num_output_; i++)
		  printf("%d:%.7f ",i+1,kernel_float_diff[i]);
	  printf("\n");
  }
}
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weights_pad(int weight_channel_offset_,int kernel_int_size,Dtype *weight){
	int kernel_max_size = kernel_shape_max_.cpu_data()[0];//kernel_int_size is before padding
	CHECK_LE(kernel_int_size+2,kernel_max_size)<<"The weight size should smaller or equal to the maximum kernel size";
	int kernel_offset = kernel_max_size*kernel_max_size;
	int kernel_num = weight_channel_offset_/kernel_offset;//5->7
	int kernel_shift=(kernel_max_size-kernel_int_size)/2;//(9-5)/2
	//printf("weight pad before:\n");
	//for(int j=0; j< 9; j++){for(int k=0; k< 9; k++){ printf("%.4e ",weight[j*9+k]);} printf("\n");}
	Dtype diff_sum=0.0;
	for (int i=0; i< kernel_num; i++){
		for(int j=0; j< kernel_max_size; j++){
		  for(int k=0; k< kernel_max_size; k++){
			  	  	  	  //1                           7
			  if(j<=kernel_shift-1||j>=kernel_max_size-kernel_shift||k<=kernel_shift-1||k>=kernel_max_size-kernel_shift){
				  weight[i*kernel_offset+j*kernel_max_size+k]=0;
			  }
			  weight[i*kernel_offset+j*kernel_max_size+kernel_shift-1]=weight[i*kernel_offset+j*kernel_max_size+kernel_shift];
			  weight[i*kernel_offset+(j+1)*kernel_max_size-kernel_shift]=weight[i*kernel_offset+(j+1)*kernel_max_size-kernel_shift-1];
		  }
		  for(int k=0;k<kernel_max_size;k++){
			  weight[i*kernel_offset+(kernel_shift-1)*kernel_max_size+k]=weight[i*kernel_offset+kernel_shift*kernel_max_size+k];
			  weight[i*kernel_offset+(kernel_max_size-kernel_shift)*kernel_max_size+k]=weight[i*kernel_offset+(kernel_max_size-kernel_shift-1)*kernel_max_size+k];
		  }
		}
	}
	for (int i=0; i< kernel_num; i++){
		for(int j=0; j< kernel_max_size; j++){
		  for(int k=0; k< kernel_max_size; k++){
			  if(((j==kernel_shift-1||j==kernel_max_size-kernel_shift)&&(k>=kernel_shift-1&&k<=kernel_max_size-kernel_shift))||
					  ((k==kernel_shift-1||k==kernel_max_size-kernel_shift)&&(j>=kernel_shift-1&&j<=kernel_max_size-kernel_shift))){
			  		diff_sum+=weight[i*kernel_offset+j*kernel_max_size+k];
			  				  //printf("%f ",weight[i*kernel_offset+j*kernel_max_size+k]);
			  }
		  }
		}
	}
	int elm_num=(kernel_int_size+2)*(kernel_int_size+2)*kernel_num;
	diff_sum=diff_sum/elm_num;
	//caffe_add_scalar(weight_channel_offset_,-diff_sum,weight);
	//printf("weight pad after:\n");
	//for(int j=0; j< 9; j++){for(int k=0; k< 9; k++){ printf("%.4e ",weight[j*9+k]);} printf("\n");}
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weights_cut(int weight_channel_offset_,int kernel_int_size,Dtype *weight){
	int kernel_max_size = kernel_shape_max_.cpu_data()[0];//kernel_int_size is after cutting
	CHECK_GE(kernel_int_size,1)<<"The weight size should be bigger or equal than 1";
	int kernel_offset = kernel_max_size*kernel_max_size;
	int kernel_num = weight_channel_offset_/kernel_offset;
	int kernel_shift=(kernel_max_size-kernel_int_size)/2;//(9-1)/2=4
	Dtype diff_sum=0.0;
	for (int i=0; i< kernel_num; i++){
		for(int j=0; j< kernel_max_size; j++){
		  for(int k=0; k< kernel_max_size; k++){
                         //3                           5                        3                            5
			  if(((j==kernel_shift-1||j==kernel_max_size-kernel_shift)&&(k>=kernel_shift-1&&k<=kernel_max_size-kernel_shift))||
					  ((k==kernel_shift-1||k==kernel_max_size-kernel_shift)&&(j>=kernel_shift-1&&j<=kernel_max_size-kernel_shift))){
				  diff_sum+=weight[i*kernel_offset+j*kernel_max_size+k];
				  //printf("%f ",weight[i*kernel_offset+j*kernel_max_size+k]);
			  }
			  if(j<=kernel_shift-1||j>=kernel_max_size-kernel_shift||k<=kernel_shift-1||k>=kernel_max_size-kernel_shift){
				  weight[i*kernel_offset+j*kernel_max_size+k]=0;
			  }
		  }
		}
	}
	int elm_num=(kernel_int_size)*(kernel_int_size)*kernel_num;
	diff_sum=diff_sum/elm_num;
	//printf("cur_diff=%f\n",diff_sum);
	//caffe_add_scalar(weight_channel_offset_,diff_sum,weight);

}


template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights_up,const Dtype* weights_down, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  //add another caffe_cpu_gemm with the
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  //printf("conv_out_channels:%d,conv_out_spatial_dim:%d,kernel_dim_down:%d, group:%d\n",conv_out_channels_,
	//	  conv_out_spatial_dim_, kernel_dim_,group_);
  for (int g = 0; g < group_; ++g) {
	//caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
	//     group_, conv_out_spatial_dim_, kernel_dim_,
	//     (Dtype)1, weights_down + weight_offset_ * g, col_buff + col_offset_ * g,
	//     (Dtype)0, output + output_offset_ * g);//with kernal size round(k)-2
	//for(int i=0; i<kernel_dim_; i++) printf("%f ",weights_down[i+kernel_dim_]);printf("\n");
	//for(int i=0; i<kernel_dim_; i++) printf("%f ",col_buff[i*conv_out_spatial_dim_+1900]);printf("\n");
	//for(int i =0; i< 1; i++) printf("%f ", output[1900]);printf("\n");
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights_up + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);//with kernal size round(k)+2
	//for(int i=0; i<kernel_dim_; i++) printf("%f ",weights_up[i+kernel_dim_]);printf("\n");
	//for(int i=0; i<kernel_dim_; i++) printf("%f ",col_buff[i*conv_out_spatial_dim_+1900]);printf("\n");
    //for(int i =0; i< 1; i++) printf("%f ", output[1900]);printf("\n");
    //the kernel_dim_, weight_offset_, col_offset_ will be different
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)down_ratio_, weights_down + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)up_ratio_, output + output_offset_ * g);//with kernal size round(k)-2
    //printf("down_ratio=%f,up_ratio=%f\n",down_ratio_,up_ratio_);
    //printf("input:");for(int i =4350; i< 4450; i++) printf("%f ", input[i]);printf("\n");
    //printf("output:");for(int i =4350; i< 4450; i++) printf("%f ", output[i]);printf("\n");
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
    const Dtype* weights_up,const Dtype* weights_down, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights_up + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)down_ratio_, weights_down + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)up_ratio_, col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);//Added by Shizhong input = 0.0*input + up_ratio * col_buff
  }
}
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights_up, Dtype* weights_down) {
  const Dtype* col_buff = input;
  //Dtype* weights_diff_up = weight_filter_up_.mutable_cpu_diff();
  //Dtype* weights_diff_down = weight_filter_down_.mutable_cpu_diff();
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();

  }
  //printf("down_ratio=%f,up_ratio=%f\n",down_ratio_,up_ratio_);
  //printf("weightin:");for(int i =100; i< 110; i++) printf("%e ", weights[i]);printf("\n");
  for (int g = 0; g < group_; ++g) {
    //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
    //    kernel_dim_, conv_out_spatial_dim_,
    //    (Dtype)up_ratio_, output + output_offset_ * g, col_buff + col_offset_ * g,
    //    (Dtype)1., weights_diff_up + weight_offset_ * g);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)up_ratio_, output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1.0, weights_up + weight_offset_ * g);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)down_ratio_, output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1.0, weights_down + weight_offset_ * g);
    //caffe_mul(weight_offset_, weight_multiplier_.cpu_data(),weights, weights);
    //caffe_scal(weight_offset_,(Dtype)1.,weights+weight_offset_ * g);
    //caffe_add(weight_filter_up_.count(),weights_diff_up,weights_diff_down,weights);
  }
  //printf("weightout:");for(int i =100; i< 110; i++) printf("%e ", weights[i]);printf("\n");
  //ReshapeFilterUpDown();
  //printf("weightout:\n");
  //for(int j=0; j< 9; j++){
//		      for(int k=0; k< 9; k++){ printf("%f ",weights[j*9+k]); }
//		      printf("\n");
 // }

}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_cpu_kernel_size(const Dtype* output_diff,const Dtype* input,
	    const Dtype* weights_up,const Dtype* weights_down, Dtype* kernel_size_diff) {
  const Dtype* col_buff = input;
	  //add another caffe_cpu_gemm with the
  Dtype* output = kernel_diff_buffer_.mutable_cpu_data();
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  //Dtype sum=0.0;
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights_up + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);//with kernal size round(k)+2
    //the kernel_dim_, weight_offset_, col_offset_ will be different
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)(-0.5), weights_down + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0.5, output + output_offset_ * g);//with kernal size round(k)-2
    //for(int i=0; i<10; i++) printf("%e ",output_diff[i]); printf("\n");
    //for(int i=0; i<10; i++) printf("%f ",output[i]); printf("\n");
    caffe_mul(output_offset_, output_diff+output_offset_ * g,output+output_offset_ * g, output+output_offset_ * g);
    //for(int i=0; i<10; i++) printf("%e ",output[i]); printf("\n");
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
              output+ output_offset_ * g, bias_multiplier_.cpu_data(), 1., kernel_size_diff);
    //for(int i=0;i<output_offset_;i++){
    //   sum+=output[i];
    //}
  }
  //kernel_size_diff[0]+=sum;
  //printf("kernl siff=%e, sum=%e\n",kernel_size_diff[0], sum);
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights_up,const Dtype* weights_down, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff=col_buffer_.gpu_data();
  }
  //printf("conv_out_channels:%d,conv_out_spatial_dim:%d, kernel_dim:%d, weight_offset:%d, col_offset:%d,output_offset:%d\n",
//		  conv_out_channels_, conv_out_spatial_dim_, kernel_dim_,
//		          weight_offset_,  col_offset_, output_offset_);
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
            group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights_up + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
            group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)down_ratio_, weights_down + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)up_ratio_, output + output_offset_ * g);
  }

}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights_up,const Dtype* weights_down, Dtype* input) {
	Dtype* col_buff = col_buffer_.mutable_gpu_data();
	if (is_1x1_) {
	    col_buff = input;
	}
	for (int g = 0; g < group_; ++g) {
	    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
	        conv_out_spatial_dim_, conv_out_channels_ / group_,
	        (Dtype)1., weights_up + weight_offset_ * g, output + output_offset_ * g,
	        (Dtype)0., col_buff + col_offset_ * g);
	    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
	        conv_out_spatial_dim_, conv_out_channels_ / group_,
	        (Dtype)down_ratio_, weights_down + weight_offset_ * g, output + output_offset_ * g,
	        (Dtype)up_ratio_, col_buff + col_offset_ * g);
	}
	if (!is_1x1_) {
	    conv_col2im_gpu(col_buff,input);//Added by Shizhong input = 0.0*input + up_ratio * col_buf
	}
}
//
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights_up, Dtype* weights_down) {//input is the bottom_data, output is the top_data
	const Dtype* col_buff = input;
	//Dtype* weights_diff_up = weight_filter_up_.mutable_gpu_diff();
	//Dtype* weights_diff_down = weight_filter_up_.mutable_gpu_diff();
	//Dtype* weights_diff_down = weight_filter_down_.mutable_gpu_diff();
	if (!is_1x1_) {
	    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
	    col_buff = col_buffer_.gpu_data();
	}
	  //printf("down_ratio=%f,up_ratio=%f\n",up_ratio,down_ratio);
	 // printf("weightin:");for(int i =100; i< 110; i++) printf("%e ", weights[i]);printf("\n");
	for (int g = 0; g < group_; ++g) {
	    //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
	    //    kernel_dim_, conv_out_spatial_dim_,
	    //    (Dtype)up_ratio_, output + output_offset_ * g, col_buff + col_offset_ * g,
	    //    (Dtype)1., weights_diff_up + weight_offset_ * g);
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
			kernel_dim_, conv_out_spatial_dim_,
			(Dtype)1, output + output_offset_ * g, col_buff + col_offset_ * g,
			(Dtype)1., weights_up + weight_offset_ * g);
		//caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
		//	kernel_dim_, conv_out_spatial_dim_,
		//	(Dtype)down_ratio_, output + output_offset_ * g, col_buff + col_offset_ * g,
		//	(Dtype)1., weights_down + weight_offset_ * g);

		//caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
		//	kernel_dim_, conv_out_spatial_dim_,
		//	(Dtype)down_ratio_, output + output_offset_ * g, col_buff + col_offset_ * g,
		//	(Dtype)1., weights_up + weight_offset_ * g);
		caffe_copy<Dtype>(weight_offset_, weights_up+weight_offset_*g, weights_down+weight_offset_*g);
		//caffe_gpu_mul(weight_offset_, weight_multiplier_.gpu_data(),weights_diff_tmp+col_offset_ * g, weights_diff_tmp+col_offset_ * g);
		//caffe_gpu_add(weight_offset_,weights_diff_tmp+col_offset_ * g,weights + weight_offset_ * g,weights + weight_offset_ * g);
	    //printf("*\n");
	   //
	    //for(int i=0; i<num_output_; i++){
	    //	caffe_gpu_dot(weight_channel_offset_,weightone_multiplier_.gpu_data(),weights+i*weight_channel_offset_,&sum);
	    //	printf("%d:%f ",i+1,sum);
	    //}
	    //printf("*\n");
	}

}
//
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {//input is the top_diff
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}
template <typename Dtype>
void BaseAdaptiveConvolutionLayer<Dtype>::backward_gpu_kernel_size(const Dtype* top_diff,const Dtype* bottom_data,
	    const Dtype* weights_up,const Dtype* weights_down,  Dtype* kernel_size_diff) {//the ouput_diff is top_diff, input is the bottom_data
  if(this->layer_param_.adaptiveconvolution_param().adaptive_term()==false){
		//kernel_size_diff[0]=0;
		return;
  }
  const Dtype* col_buff = bottom_data;
  //Dtype* weights_up = weight_ratio_up_.mutable_gpu_data();
  //Dtype* weights_down = weight_ratio_down_.mutable_gpu_data();
	  //add another caffe_cpu_gemm with the
  Dtype* output = kernel_diff_buffer_.mutable_gpu_data();
  //printf("diff down_ratio=%f,diff up_ratio=%f\n",down_ratio,up_ratio);
  if (!is_1x1_) {
    conv_im2col_gpu(bottom_data, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights_up + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);//with kernal size round(k)+2
    //the kernel_dim_, weight_offset_, col_offset_ will be different
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)(-0.5), weights_down + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0.5, output + output_offset_ * g);//with kernal size round(k)-2
    //for(int i=0; i<10; i++) printf("%e ",output_diff[i]); printf("\n");
    //for(int i=0; i<10; i++) printf("%f ",output[i]); printf("\n");
    caffe_gpu_mul(output_offset_, top_diff,output+ output_offset_ * g, output+output_offset_ * g);
    //caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
    //      output+ output_offset_ * g, bias_multiplier_.gpu_data(), 1., kernel_size_diff);
    Dtype sum=0.0;
    caffe_gpu_dot(output_offset_,output_multiplier_.gpu_data(),output+output_offset_ * g,&sum);
    //printf("cur_diff=%f\n",sum);
    caffe_gpu_add_scalar(num_output_,sum,kernel_size_diff);
  }
 // kernel_size_diff[0]+=sum;
  //printf("kernl siff=%f, sum=%f\n",kernel_size_diff[0], sum);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseAdaptiveConvolutionLayer);

}  // namespace caffe
