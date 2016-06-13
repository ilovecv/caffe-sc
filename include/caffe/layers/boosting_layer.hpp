#ifndef CAFFE_BOOSTING_LAYER_HPP_
#define CAFFE_BOOSTING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief a boosting layer, select the feature from previous input layer and make decision based on the learned strong classifiers
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto parameters.
 */
template <typename Dtype>
class BoostingLayer : public Layer<Dtype> {
 public:
  explicit BoostingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Boosting"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }//the input data and label
  virtual inline int ExactNumTopBlobs() const { return 3; }//the strong classifier score, weak classifier score and weak classifier label

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //in the forward process, this function used to select the weak classifier and construct the strong classifier.
  virtual void adaboost(const Dtype* bottom_data, const Dtype* bottom_label, Dtype* alpha, Dtype* thresh, Dtype* sign);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;// Input sample number
  int K_;// The input data dimension
  int N_;// The selected weak classifier number
  float imith_;//The parameter to control the shape of the sign function
  float imithrate_;// the rate of imith value imith_=imithrate_/3
  string bottomdatafile_;//For debug: filename to write the bottom data
  string weakclassifierfile_;//For debug: filename to write the weak classifier
  string bottomdifffile_;//For debug: filename to write the bootom diff data
  string strongscorefile_;//For debug: to write the strong classifier score
  string weakscorefile_;//For debug: to write the weak classifier score
  string strongdifffile_;//For debug: to write the strong diff
  string weakdifffile_;////For debug: to write the weak diff
  string strongratefile_;//For debug: to write the strong classifier rate
  string classifiernumfile_;
  string strongalphafile_;
  bool debug_;
  int ite_;// the iteration number for CNN
};


}  // namespace caffe

#endif  // CAFFE_BOOSTING_LAYER_HPP_
