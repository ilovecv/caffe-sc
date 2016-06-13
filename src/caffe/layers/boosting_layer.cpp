/*
 * boosting_layer.cpp
 *
 *  Created on: Mar 23, 2016
 *      Author: handong
 */
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/boosting_layer.hpp"
#include "caffe/util/adaBoostBinary.hpp"

namespace caffe {


template <typename Dtype>
void BoostingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int most_weak_num = this->layer_param_.boosting_param().num_output();

	imithrate_=this->layer_param_.boosting_param().imith();
	N_ = most_weak_num;

	const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.boosting_param().axis());
	// Dimensions starting from "axis" are "flattened" into a single
	// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
	// and axis == 1, N inner products with dimension CHW are performed.
	K_ = bottom[0]->count(axis);
	M_ = bottom[0]->count(0, axis);
	this->blobs_.resize(3);
	vector<int> para_shape(1, K_);
	this->blobs_[0].reset(new Blob<Dtype>(para_shape));//to save alpha, alpha is the weight for each weak classifier
	this->blobs_[1].reset(new Blob<Dtype>(para_shape));//to save thresh, thresh is the thresh for each weak classifier
	this->blobs_[2].reset(new Blob<Dtype>(para_shape));//to save sign, sign is the sign for each weak classifier used to keep the mean positive is bigger than the mean negatiave
	ite_=0;
	debug_=false;
	//if(debug_){
	string dir("/media/datadisk/database/FERA2015/data_steps/testResult/normWarpImageData/");
	classifiernumfile_=dir+"classifiernum.txt";
	strongalphafile_=dir+"strongalpha.txt";
	//}

}

template <typename Dtype>
void BoostingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Figure out the dimensions
	const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.boosting_param().axis());
	const int new_K = bottom[0]->count(axis);
	CHECK_EQ(K_, new_K)
	<< "Input size incompatible with boosting parameters.";
	// The first "axis" dimensions are independent inner products; the total
	// number of these is M_, the product over these dimensions.

	// The top shape will be the bottom shape with the flattened axes dropped,
	// and replaced by a single axis with dimension num_output (N_).
	vector<int> top_shape(2,M_);
	top_shape[1]=1;
	top[0]->Reshape(top_shape);//for alpha
	vector<int> weak_shape(2,M_);
	weak_shape[1]=N_;
	top[1]->Reshape(weak_shape);//for thresh
	top[2]->Reshape(weak_shape);//for sign
	// Set up the bias multiplier
	if(debug_){
		ostringstream ss;
		ss << ite_;
		string dir("/media/datadisk/database/FERA2015/data_steps/testResult/normWarpImageData/");
		bottomdatafile_= dir + "bottomdata" + ss.str() + ".txt";
		weakclassifierfile_=dir+"weakclassifier"+ss.str()+".txt";
		bottomdifffile_=dir+"backdiff"+ss.str()+".txt";
		strongscorefile_=dir+"strongscore"+ss.str()+".txt";
		weakscorefile_=dir+"weakscore"+ss.str()+".txt";
		strongdifffile_=dir+"strongdiff"+ss.str()+".txt";
		weakdifffile_=dir+"weakdiff"+ss.str()+".txt";
		strongratefile_=dir+"strongrate.txt";
	}

	ite_++;

}
/**
 * Test the AdaBoost strong classifier on training set
 */
template <typename Dtype>
void testdata(const Dtype* inputdata, Dtype* top_strong_score,Dtype* alpha,Dtype* thresh,Dtype* sign , int samplenum, int dimension, double imith)
{
	for(int i=0; i<samplenum; i++)
	{
		top_strong_score[i]=0.0;
		for(int j=0; j<dimension; j++)
		{
			if(sign[j]!=0){//only calculate the selected weak classifier

				double curscore=inputdata[i*dimension+j]*sign[j]-thresh[j];
				double curweak=curscore/sqrt(curscore*curscore+imith*imith);
				top_strong_score[i]+=alpha[j]*curweak;
			}
		}
		//transfer the data from [-1 1] to [0 1]
		top_strong_score[i]=(top_strong_score[i]+1)/2;
	}

}
/**
 * In the forward process, this function used to select the weak classifier and construct the strong classifier.
 */
template <typename Dtype>
void BoostingLayer<Dtype>::adaboost(const Dtype* bottom_data, const Dtype* bottom_label, Dtype* alpha, Dtype* thresh, Dtype* sign)
{
	adaBoostBinary  adaboost;
	bool *maplabel=(bool *)malloc(sizeof(bool)*K_);
	for (int i=0; i<K_; i++) maplabel[i]=false;
	int posnum=0;
	int negnum=0;
	for(int i=0; i<M_; i++)
	{
		if(bottom_label[i]==0)negnum++;
		else posnum++;
	}
	//printf("posnum: %d, negnum:%d dimension:%d \n", posnum, negnum, K_);
	adaboost.allocateMemory(posnum,negnum,N_);
	//prepare the train data
	CvMat *feature_train = cvCreateMat(posnum+negnum, K_, CV_32FC1);
	CvMat *labelMat = cvCreateMat(posnum+negnum,1,CV_8UC1);
	for(int i=0; i<M_; i++){
		for(int j=0; j<K_; j++){
			feature_train->data.fl[i*K_+j]=bottom_data[i*K_+j];
		}
		labelMat->data.ptr[i]=bottom_label[i];
	}
	adaboost.Initialize(labelMat);
	float sum_alpha=0.0;
	float min_error=0.0, errorrate_ith_iter=0.0;
	std::list<double> first_list;
	std::list<double> second_list;
	//double check_rate=0.05;
    int listnum=4;
    //train AdaBoost and select features
	for(int i=0; i< N_; i++)
	{
		lda_classifier current_best_classifier;
		current_best_classifier.minerror=-10;
		//adaboost.choosefeature_fast(feature_train,labelMat, &current_best_classifier, maplabel,imith_,i);
		adaboost.choosefeature_fastupdate(feature_train,labelMat, &current_best_classifier,maplabel, sign,thresh,imith_,i);//feature selection
		adaboost.updataWeight(feature_train, &current_best_classifier,imith_, i);//update the weight
		sum_alpha+=current_best_classifier.alpha; //calculate the summation of the alpha
		errorrate_ith_iter = adaboost.testStrongClassifier();//calculate the error rate in the training data
		first_list.push_front(errorrate_ith_iter);
		double first_sum=0.0, second_sum=0.0;
		if(i>=listnum){
			second_list.push_front(first_list.back());
		    first_list.pop_back();
			if(i>=2*listnum)
			{
				second_list.pop_back();
		        first_sum=std::accumulate(first_list.begin(), first_list.end(),0.0f);
		        second_sum=std::accumulate(second_list.begin(), second_list.end(),0.0f);
		        if(second_sum<=first_sum)
		        {
		        	printf("Break, the error in the training set begin to decrease\n");
		            break;
		        }
		    }
		}
		min_error=current_best_classifier.minerror;
		//printf("Strong classifier error: %f, weak classifier error: %f \n", errorrate_ith_iter, min_error);
		if (min_error <= 0)
		{
			printf("Break: the current iteration error is bigger than 0.5\n");
			break;
		}
		else if (errorrate_ith_iter <= 0.03)
		{
			printf("Break: We have got the best performance, 99%% strong classifier performance!\n");
			break;
		}
	}
	int cur_weaknum=adaboost.strongclassifier.size();
	//double sum1=0,sum2=0;
	for(int i=0; i<K_; i++){
		if(sign[i]!=0) alpha[i]=alpha[i]*(ite_-2)/(ite_-1);
	//	sum1+=alpha[i];
	}
	printf("iteration: %d ===========\n",cur_weaknum);
	//printf("Strong classifier error: %f, weak classifier error: %f \n", errorrate_ith_iter, min_error);
	if(debug_&&ite_<10){
		std::ofstream outstrongrate(strongratefile_.c_str(), ios::app);
		if(!(outstrongrate.is_open())){
			std::cout << "Can not open the weak classifier file\n";
		}
		outstrongrate << errorrate_ith_iter <<" " << cur_weaknum << "\n";
		outstrongrate.close();
	}
	//printf("iter=%d\n\n",ite_);
	for(int i=0; i<cur_weaknum; i++){
		int curbindex=adaboost.strongclassifier[i].bindex;
		double curalpha=adaboost.strongclassifier[i].alpha/sum_alpha;
		alpha[curbindex]+=curalpha/(ite_-1);
	//	sum2+=curalpha/(ite_-1);
		if(sign[curbindex]==0){
			thresh[curbindex]=adaboost.strongclassifier[i].pro_thresh;
			sign[curbindex]=adaboost.strongclassifier[i].thr_sign;
		}
	}
	int totalnum=0;
	for(int i=0; i<K_;i++){
		if(sign[i]!=0) totalnum++;
	}
	if(debug_==false){
		std::ofstream outclassifiernum(classifiernumfile_.c_str(), ios::app);
		if(!(outclassifiernum.is_open())){
	      	std::cout << "Can not open the classifier_num file\n";
		}
		outclassifiernum <<ite_ <<" "<< totalnum << " " << cur_weaknum<<"\n";
		outclassifiernum.close();
	}
	if(debug_==false){
		std::ofstream strongalpha(strongalphafile_.c_str(),ios::app);
		if(!(strongalpha.is_open())){
			std::cout << "Can not open the strong alpha file\n";
		}
		strongalpha<<ite_ <<" ";
		for(int i=0; i<K_; i++){
			strongalpha << alpha[i]<<" ";
		}
		strongalpha<<"\n";
		strongalpha.close();
	}
	//printf("sum1=%f,sum2=%f\n",sum1,sum2);
	//write the weak classifier
	if(debug_&&ite_<=10){
    std::ofstream outweakclassifier(weakclassifierfile_.c_str(), ios::out);
    if(!(outweakclassifier.is_open())){
      	std::cout << "Can not open the weak classifier file\n";
    }
    for(int i=0; i<cur_weaknum; i++){
    	outweakclassifier <<
    			adaboost.strongclassifier[i].bindex <<" "<<
    			adaboost.strongclassifier[i].alpha<< " " <<
    			adaboost.strongclassifier[i].pro_thresh << " "<<
    			adaboost.strongclassifier[i].thr_sign<<"\n";
    }
    	outweakclassifier.close();
    }
	//return the weight and top_data
	//realease the space
	cvRelease((void **)&feature_train);
	cvRelease((void **)&labelMat);
	free(maplabel);
}
template <typename Dtype>
void BoostingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	Dtype* top_strong_score = top[0]->mutable_cpu_data();
	Dtype* top_weak_score = top[1]->mutable_cpu_data();
	Dtype* top_label = top[2]->mutable_cpu_data();
	Dtype *alpha = this->blobs_[0]->mutable_cpu_data();
	Dtype *thresh = this->blobs_[1]->mutable_cpu_data();
	Dtype *sign = this->blobs_[2]->mutable_cpu_data();
	caffe_set(top[0]->count(), Dtype(0), top_strong_score);
	caffe_set(top[1]->count(), Dtype(0), top_weak_score);
	caffe_set(top[2]->count(), Dtype(0), top_label);
	double curstd=StDev(bottom_data, bottom[0]->count());
	//printf("std=%f\n",curstd);
	imith_=imithrate_;
    //for(int i=0; i< 10; i++) printf("%f ",bottom_data[i]);
    //printf("\n");
    //for(int i=0; i< 10; i++) printf("%f ",alpha[i]);
    //printf("\n");
	if(this->phase_ == TRAIN){
		adaboost(bottom_data, bottom_label,alpha, thresh, sign);
	}
	testdata(bottom_data,top_strong_score,alpha, thresh, sign, M_,K_, imith_);
	//printf("imith=%f\n",imith_);
	for(int i=0; i< M_; i++){
		int k=0;
		for(int j=0; j<K_; j++){
			if(sign[j]!=0){
				double x=bottom_data[i*K_+j]*sign[j]-thresh[j];
				top_weak_score[i*N_+k]=(x/sqrt(x*x+imith_*imith_)+1)/2;
				//if(i==0)
				//printf("%f %f *",x,top_weak_score[i*N_+k]);
				//if(x>=0)
				//top_weak_score[i*N_+k]=1;
				//else
				//	top_weak_score[i*N_+k]=-1;
				top_label[i*N_+k]=bottom_label[i];
				k++;
			}
		}
	}
	//write the data
	if(this->phase_ == TRAIN&&ite_<=10&&debug_){
		//bottomdata
		std::ofstream outbottomdata(bottomdatafile_.c_str(), ios::out);
		if(!(outbottomdata.is_open())){
		  	std::cout << "Can not open the file\n";
		}
		for(int i=0; i<M_; i++)
		{
			for(int j=0; j<K_; j++)
		  	{
		 		outbottomdata<<bottom_data[i*K_+j]<<" ";
	    	}
		 	outbottomdata<<bottom_label[i]<<"\n";
	    }
		outbottomdata.close();
		//weakscore
		std::ofstream outweakscore(weakscorefile_.c_str(), ios::out);
		if(!(outweakscore.is_open())){
		  	std::cout << "Can not open the file\n";
		}
		for(int i=0; i<M_; i++)
		{
			for(int j=0; j<N_; j++)
		  	{
		 		outweakscore<<top_weak_score[i*N_+j]<<" ";
	    	}
		 	outweakscore<<bottom_label[i]<<"\n";
	    }
		outweakscore.close();
		//strongscore
		std::ofstream outstrongscore(strongscorefile_.c_str(), ios::out);
		if(!(outstrongscore.is_open())){
		  	std::cout << "Can not open the file\n";
		}
		for(int i=0; i<M_; i++)
		{
			outstrongscore<<top_strong_score[i]<<" "<<bottom_label[i]<<"\n";
	    }
		outstrongscore.close();
	}

	//printf("New Iteration----------------------------------------------\n");
	//printf("alpha data-----------\n");
	//for(int i=0; i<K_; i++) printf("%f ", alpha[i]);
	//printf("\n");
	//printf("The bottom data--------------\n");
	//for(int i=0; i<10; i++){
	//	printf("%f ", bottom_data[i]);
	///}
	//printf("\n");
	//printf("The top data\n");
	//for(int i=0; i<10; i++){
	//		printf("%f ", top_data[i]);
	//}
	//printf("\n");

}

template <typename Dtype>
void BoostingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_strong_diff = top[0]->cpu_diff();
	const Dtype* top_weak_diff = top[1]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* thresh_diff = this->blobs_[1]->mutable_cpu_diff();
	Dtype* alpha_diff= this->blobs_[0]->mutable_cpu_diff();
	caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype *alpha = this->blobs_[0]->cpu_data();
	const Dtype *thresh = this->blobs_[1]->cpu_data();
	const Dtype *sign = this->blobs_[2]->cpu_data();
	caffe_set(K_, Dtype(0), thresh_diff);
	caffe_set(K_, Dtype(0), alpha_diff);
	int k=0;
	for (int i=0; i<K_; i++){
		if(sign[i]!=0) k++;
	}
    int weaknum=k;
	for(int i=0; i< M_; i++)
	{
		int k=0;
		for(int j=0; j< K_; j++)
		{
			if(sign[j]>=0.5||sign[j]<=-0.5)
			{
				double curscore=bottom_data[i*K_+j]*sign[j]-thresh[j];
				double weakdiff=top_weak_diff[i*N_+k]*imith_*imith_/pow(imith_*imith_+curscore*curscore,1.5);
				double strongdiff=top_strong_diff[i]*weaknum*alpha[j]*imith_*imith_/pow(imith_*imith_+curscore*curscore,1.5);
				double combinediff=(weakdiff+strongdiff)/2;
				//double alphadiff=top_strong_diff[i]*curscore/sqrt(curscore*curscore+imith_*imith_);
				thresh_diff[j]+=-combinediff;
				//alpha_diff[j]+=alphadiff;
				bottom_diff[i*K_+j]=sign[j]*combinediff;
				//if(i==0&&j<=5)printf("%e %f %f*", bottom_diff[i*K_+j], top_strong_diff[i], imith_);
				//if(i==0&&j<=10)printf("%f %f*", imith_*imith_/pow(imith_*imith_+curscore*curscore,1.5),curscore);

				k++;
			}
		}
	}
	//printf("\n");
	//write the bottom diff
	if(ite_<=10&&debug_){
		std::ofstream outbottomdiff(bottomdifffile_.c_str(), ios::out);
		if(!(outbottomdiff.is_open())){
			std::cout << "Can not open the bottom diff file\n";
		}
		for(int i=0; i<M_; i++)
		{
			for(int j=0; j<K_; j++)
			{
				outbottomdiff<<bottom_diff[i*K_+j]<<" ";
			}
			outbottomdiff<<"\n";
		}
		outbottomdiff.close();
		//top_weak_diff
		std::ofstream outweakdiff(weakdifffile_.c_str(), ios::out);
		if(!(outweakdiff.is_open())){
			std::cout << "Can not open the bottom diff file\n";
		}
		for(int i=0; i<M_; i++)
		{
			for(int j=0; j<N_; j++)
			{
				outweakdiff<<top_weak_diff[i*N_+j]<<" ";
			}
			outweakdiff<<"\n";
		}
		outweakdiff.close();
		//top_strong_diff
		std::ofstream outstrongdiff(strongdifffile_.c_str(), ios::out);
		if(!(outstrongdiff.is_open())){
			std::cout << "Can not open the bottom diff file\n";
		}
		for(int i=0; i<M_; i++)
		{

			outstrongdiff<<top_strong_diff[i]<<" ";
			outstrongdiff<<"\n";
		}
		outstrongdiff.close();
	}
	//printf("The bottom diff--------------\n");
	//for(int i=0; i<10; i++){
	//	printf("%f ", bottom_data[i]);
	//}
	//printf("||");
	//for(int i=0; i<10; i++){
	//	printf("%f ", thresh[i]);
	//}
    //printf("\n");
	/*printf("The sign value--------------\n");
    for(int i=0; i<K_; i++){
		printf("%f ", sign[i]);
	}
	printf("The bottom diff--------------\n");
	for(int i=0; i<100; i++){
		printf("%f ", top_strong_diff[i]);
	}
	printf("\n");
	printf("The top diff\n");
	for(int i=0; i<100; i++){
		printf("%f ", bottom_diff[i]);
	}
	printf("\n");
	// Gradient with respect to bottom data
    //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 1, (Dtype)1.,
    //top_diff, diff_, (Dtype)0., bottom_diff);
    double maxnum=0, minnum=0;
    double avgnew = han_avg(top_diff, top[0]->count(), maxnum, minnum);
    printf("avg=%f, maxnum=%f, minnum=%f\n",avgnew, maxnum, minnum);

    avgnew = han_avg(bottom_diff, bottom[0]->count(), maxnum, minnum);
    printf("avg=%f, maxnum=%f, minnum=%f\n",avgnew, maxnum, minnum);*/
}
#ifdef CPU_ONLY
STUB_GPU(BoostingLayer);
#endif

INSTANTIATE_CLASS(BoostingLayer);
REGISTER_LAYER_CLASS(Boosting);

}  // namespace caffe




