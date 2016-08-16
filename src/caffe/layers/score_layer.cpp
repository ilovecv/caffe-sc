/*
 * score_layer.cpp
 *
 *  Created on: Mar 22, 2016
 *      Author: handong
 */
#include <functional>
#include <utility>
#include <vector>
#include <cstdio>


#include "caffe/layers/score_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScoreLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.has_score_param()){
      file_name =this->layer_param_.score_param().dest_file().c_str();
  }
}

template <typename Dtype>
void ScoreLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  //CHECK_EQ(bottom[1]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void ScoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int numdata = bottom[0]->num();
  int dimdata=bottom[0]->count() / numdata;
  int numlabel=bottom[1]->num();
  int dimlabel = bottom[1]->count() / bottom[1]->num();
  int dimodd=dimdata/dimlabel;
  //printf("dimdata=%d, dimlabel=%d,dimodd=%d\n",dimdata, dimlabel, dimodd);
  if(this->layer_param_.has_score_param()){
     //std::ofstream outfile(file_name, ios::app);
	 FILE* fout = fopen(file_name, "a");
     //std::cout << writefile_ <<endl;
     for (int i = 0; i < numlabel; ++i) {
    // Top-k accuracy
    	 for (int j = 0; j < dimlabel; ++j) {
    		 for(int k=0; k<dimodd; k++){
    			 fprintf(fout, "%g ",bottom_data[i*dimdata+k+j]);
    		 }
    		 fprintf(fout, "%d ",(int)(bottom_label[i*dimlabel+j]));
         //outfile << bottom_data[i*dim+j] << " ";
       }
       fprintf(fout, "\n");
       //outfile << bottom_label[i] << "\n";

     }
     fclose(fout);
     //outfile.close();
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(ScoreLayer);
REGISTER_LAYER_CLASS(Score);

}




