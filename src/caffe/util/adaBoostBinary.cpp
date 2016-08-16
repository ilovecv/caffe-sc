/**
 *The AdaBoost algorithm file
 *@author Shizhong Han
 *@verion 1.0 03/11/2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include "caffe/util/adaBoostBinary.hpp"
#define SIZE 200
/**
 *The default constructor
 */
namespace caffe{
/**
 *The AdaBoost algorithm file
 *@author Shizhong Han
 *@verion 1.3 03/11/2015
 */
/**
 *The constructor
 */


adaBoostBinary::adaBoostBinary()
{
	weight=NULL;
	iter_res=NULL;
	labelsign = NULL;
	maxIterationNum=0;
	pos_sample_num=0;
	neg_sample_num=0;
	rate_margin=0.01;

}
/**
 *  deconstructor
 */
adaBoostBinary::~adaBoostBinary()
{
	cvRelease((void **)&weight);
	free(labelsign);
	free(iter_res);
}
/**
 *call the memory for the AdaBoost variable
 *@param posnum The positive sample number
 *@param negnum The negative sample number
 *@param iterNum The maximum iteration number
 *@return 1 If it is successful
 */
int adaBoostBinary::allocateMemory(int posnum, int negnum, int iterNum)
{
	pos_sample_num=posnum;
	neg_sample_num=negnum;
	maxIterationNum=iterNum;
	int sample_num=pos_sample_num+neg_sample_num;
	if(pos_sample_num*neg_sample_num*maxIterationNum==0)
	{
		printf("The value of positive sample number and negative sample number and iteration number should not be zero\n");
		return 0;
	}
	weight=cvCreateMat(sample_num, 1, CV_32FC1);
	iter_res=(float*)malloc(sizeof(float)*sample_num*maxIterationNum);
	labelsign=(int *)malloc(sizeof(int)*sample_num);
	return 1;
}
/**
*Initialize the training label and transfer the label from 0 1 to short -1 and 1
*@param label The training label of 0 and 1 value
*@return 1 If it is successful
*/
int adaBoostBinary::Initialize(CvMat *label)
{
	int sample_num = pos_sample_num + neg_sample_num;
	InitializeWeightFile(label);
	//initialize the label sign
	for(int i=0; i< sample_num; i++)
	{
		if(label->data.ptr[i]>0)
			labelsign[i]=1;
		else
			labelsign[i]=-1;
	}
	return 1;
}
/**
*Initialize weight matrix using the summation value of sum
*@param label The training label of 0 and 1 value
*@return 1 If it is successful
*/
int adaBoostBinary::InitializeWeightFile(CvMat *label)
{
	int sample_num=pos_sample_num+neg_sample_num;
	float weight_pos = 0.5f/pos_sample_num;
	float weight_neg = 0.5f/neg_sample_num;
	for(int i = 0; i < sample_num; i++)
	{
		if(label->data.ptr[i]>0)
			cvmSet(weight, i, 0, weight_pos);
		else
			cvmSet(weight, i, 0, weight_neg);
	}
	return 1;

}
/**
 *Choose the weak classifier selection process for the Convolutional Neural Network
 *@param data The positive and negative training samples
 *@param label The ground truth label of the training samples
 *@param current_best_classifier The current weak classifier for selection.
 *@param bindexlabel The BOOL value to ensure that each feature can only be choose once
 *@param sign The history sign for each weak classifier 0 1 -1. 0 means that it's selected in the history. 1 means mean value of positive is bigger than negative
 *@param thresh The history threshold for each weak classifier for Convolutional Neural Network
 *@param imith To control the shape of the sign function
 *@param curIteration The current iteration number of strong classifier
 */

template <typename Dtype>
int adaBoostBinary::choosefeature_fastupdate(CvMat *data,CvMat *label, lda_classifier *current_best_classifier, bool* bindexlabel,Dtype *sign,Dtype *thresh, float imith, int curIteration)
{
	int sample_num=pos_sample_num+neg_sample_num;
	int size_dim=data->cols;
	int paraNum=1;
	int numInpara=size_dim/paraNum;
	lda_classifier *para_best_classifier = (lda_classifier*)malloc(sizeof(lda_classifier)*paraNum);
	CvMat *labelneg=cvCreateMat(sample_num, 1, CV_8UC1);
	for(int i=0; i<sample_num; i++){
		if(label->data.ptr[i]>0){
			labelneg->data.ptr[i]=0;
		}
		else{
			labelneg->data.ptr[i]=1;
		}
	}
	float *para_res=(float*)malloc(sizeof(float)*sample_num*paraNum);
	//#pragma omp parallel for
	for(int j1=0; j1<paraNum; j1++)
	{
		para_best_classifier[j1].minerror=-10;
		for(int j2=0; j2<numInpara; j2++)
		{
			int j=j1*numInpara+j2;
			// if((j+1)%8<=1) continue;
			CvMat *score=cvCreateMat(sample_num,1,CV_32FC1);
			///find the current best error and threshold
			CvScalar meanpos, meanneg, stdpos, stdneg;
			for(int t=0; t<sample_num; t++) score->data.fl[t] = cvmGet(data, t, j);
			//printf("arow=%d, acol=%d, brow=%d, bcol=%d\n",score->rows,score->cols,label->rows, label->cols);
			float cur_error=0.0;
			int thr_sign=1;
			float threshold=0.0;
			if(sign[j]==0){
				cvAvgSdv(score, &meanpos, &stdpos, label);
				cvAvgSdv(score, &meanneg, &stdneg, labelneg);
				if(stdpos.val[0]+stdneg.val[0]>0.0) threshold = (meanpos.val[0]*stdneg.val[0]+meanneg.val[0]*stdpos.val[0])/(stdpos.val[0]+stdneg.val[0]);
				else threshold = (meanpos.val[0]+meanneg.val[0])/2;
				if(threshold==0.0) continue;
				if(meanpos.val[0]<meanneg.val[0])
				{
					thr_sign=-1;
					threshold=-threshold;
				}
			}
			else{
				thr_sign=sign[j];
				threshold=thresh[j];
			}
			for(int t=0; t<sample_num; t++)
			{
				float curscore=score->data.fl[t]*thr_sign-threshold;
				float htx=curscore/sqrt(curscore*curscore+imith*imith);
				cur_error+=weight->data.fl[t]*htx*labelsign[t];
			}
			if (cur_error> para_best_classifier[j1].minerror&&bindexlabel[j]==false)
			{
				para_best_classifier[j1].minerror = cur_error;
				//printf("*%f ",cur_error);
				para_best_classifier[j1].bindex = j;
				para_best_classifier[j1].pro_thresh = threshold;
				para_best_classifier[j1].thr_sign = thr_sign;
				// float check_error=0.0f;
				for(int t=0;t<sample_num; t++)
				{
					float curscore=score->data.fl[t]*thr_sign-threshold;
					float htx=curscore/sqrt(curscore*curscore+imith*imith);
					//para_res[j1*sample_num+t]=curscore/sqrt(curscore*curscore+imith*imith);
					para_res[j1*sample_num+t]=htx;
				}
			}
			cvRelease((void **)&score);

		}//for para inter pixle j2
	}//for para outer pixel j1
	for(int i=0; i<paraNum; i++)
	{
		if(current_best_classifier->minerror<para_best_classifier[i].minerror)
		{
			current_best_classifier->minerror=para_best_classifier[i].minerror;
			current_best_classifier->bindex=para_best_classifier[i].bindex;
			current_best_classifier->pro_thresh=para_best_classifier[i].pro_thresh;
			current_best_classifier->thr_sign=para_best_classifier[i].thr_sign;
			for(int t=0; t<sample_num; t++) iter_res[curIteration*sample_num+t]=para_res[i*sample_num+t];
		}
	}
	bindexlabel[current_best_classifier->bindex]=true;
	cvRelease((void **)&labelneg);
	free(para_res);
	free(para_best_classifier);
	return 1;
}

template int adaBoostBinary::choosefeature_fastupdate<float>(CvMat *data,CvMat *label, lda_classifier *current_best_classifier, bool* bindexlabel,float *sign,float *thresh, float imith, int curIteration);
template int adaBoostBinary::choosefeature_fastupdate<double>(CvMat *data,CvMat *label, lda_classifier *current_best_classifier, bool* bindexlabel,double *sign,double *thresh, float imith, int curIteration);
/**
 *Choose the weak classifier using the fast method just based on the distribution of the weight
 *@param data The positive and negative training samples
 *@param label the ground truth label of the training samples
 *@param current_best_classifier The current weak classifier for selection.
 *@param bindexlabel The BOOL value to ensure that each feature can only be choose once
 *@param imith To control the shape of the sign function
 *@param curIteration The current iteration number of strong classifier
 */
int adaBoostBinary::choosefeature_fast(CvMat *data, CvMat *label, lda_classifier *current_best_classifier,bool *bindexlabel, float imith, int curIteration)
{
	int sample_num=pos_sample_num+neg_sample_num;
	int size_dim=data->cols;
	int paraNum=1;
	int numInpara=size_dim/paraNum;
	lda_classifier *para_best_classifier = (lda_classifier*)malloc(sizeof(lda_classifier)*paraNum);
	CvMat *labelneg=cvCreateMat(sample_num, 1, CV_8UC1);
	for(int i=0; i<sample_num; i++){
		if(label->data.ptr[i]>0){
			labelneg->data.ptr[i]=0;
		}
		else{
			labelneg->data.ptr[i]=1;
		}
	}
	float *para_res=(float*)malloc(sizeof(float)*sample_num*paraNum);
	//#pragma omp parallel for
	for(int j1=0; j1<paraNum; j1++)
	{
		para_best_classifier[j1].minerror=-10;
		for(int j2=0; j2<numInpara; j2++)
		{
			int j=j1*numInpara+j2;
			// if((j+1)%8<=1) continue;
			CvMat *score=cvCreateMat(sample_num,1,CV_32FC1);
			///find the current best error and threshold
			CvScalar meanpos, meanneg, stdpos, stdneg;
			for(int t=0; t<sample_num; t++) score->data.fl[t] = cvmGet(data, t, j);
			//printf("arow=%d, acol=%d, brow=%d, bcol=%d\n",score->rows,score->cols,label->rows, label->cols);
			cvAvgSdv(score, &meanpos, &stdpos, label);
			cvAvgSdv(score, &meanneg, &stdneg, labelneg);

			float threshold=0.0;
			if(stdpos.val[0]+stdneg.val[0]>0.0) threshold = (meanpos.val[0]*stdneg.val[0]+meanneg.val[0]*stdpos.val[0])/(stdpos.val[0]+stdneg.val[0]);
			else threshold = (meanpos.val[0]+meanneg.val[0])/2;
			if(threshold==0.0) continue;
			float cur_error=0.0;
			int thr_sign=1;
			if(meanpos.val[0]<meanneg.val[0])
			{
				thr_sign=-1;
				threshold=-threshold;
			}
			for(int t=0; t<sample_num; t++)
			{
				float curscore=score->data.fl[t]*thr_sign-threshold;
				float htx=curscore/sqrt(curscore*curscore+imith*imith);
				cur_error+=weight->data.fl[t]*htx*labelsign[t];

			}
			if (cur_error> para_best_classifier[j1].minerror&&bindexlabel[j]==false)
			{
				para_best_classifier[j1].minerror = cur_error;
				printf("*%f ",cur_error);
				para_best_classifier[j1].bindex = j;
				para_best_classifier[j1].pro_thresh = threshold;
				para_best_classifier[j1].thr_sign = thr_sign;

				// float check_error=0.0f;
				for(int t=0;t<sample_num; t++)
				{
					float curscore=score->data.fl[t]*thr_sign-threshold;
					float htx=curscore/sqrt(curscore*curscore+imith*imith);
					//para_res[j1*sample_num+t]=curscore/sqrt(curscore*curscore+imith*imith);
					para_res[j1*sample_num+t]=htx;
				}
			}
			cvRelease((void **)&score);

		}//for para inter pixle j2
	}//for para outer pixel j1
	for(int i=0; i<paraNum; i++)
	{
		if(current_best_classifier->minerror>para_best_classifier[i].minerror)
		{
			current_best_classifier->minerror=para_best_classifier[i].minerror;
			current_best_classifier->bindex=para_best_classifier[i].bindex;
			current_best_classifier->pro_thresh=para_best_classifier[i].pro_thresh;
			current_best_classifier->thr_sign=para_best_classifier[i].thr_sign;

			for(int t=0; t<sample_num; t++) iter_res[curIteration*sample_num+t]=para_res[i*sample_num+t];
		}
	}
	bindexlabel[current_best_classifier->bindex]=true;
	cvRelease((void **)&labelneg);
	free(para_res);
	free(para_best_classifier);
	return 1;
}
/**
 *Load the feature continually
 *@param feature_train The positive and negative training feature
 *@param trainfile The file path for reading the train feature
 *@param ib The begining row in the feature_train for current data reading from trainfile. It's used when the data is saved in multiple files.
 *@param id The ending row in the feature_train for current data reading from trainfile
 *@param dim is the dimension of the training feature
 *@param inum The current set number if the data is divided into multiple set and saved in multiple files
 */
int adaBoostBinary::LoadDataAllFeat_set(CvMat *feature_train, const char* trainfile,int ib, int id,int dim,int &inum)
{
	int alldim=feature_train->cols;
	FILE *fptrain;
	fptrain=fopen(trainfile,"r");
	if(NULL==fptrain)
	{
		printf("ERROR: Open train file %s failed\n",trainfile);
		fclose(fptrain);
		return -1;
	}
	float tmpfeat;
	for(int j=ib; j < id; j++)
	{
		for(int l=0; l<dim; l++)
		{
			if(EOF==fscanf(fptrain, "%f ", &tmpfeat))
			{
				printf("ERROR: the file %s is not completed\n", trainfile);
				return -1;
			}
			else
			{
				feature_train->data.fl[inum*alldim+l]=tmpfeat;
			}
		}
		if(inum==feature_train->rows)
		{
			printf("ERROR: the size of train data in the file %s is more than the designed size %d\n", trainfile, inum);
			return -1;
		}
		inum++;

	}
	printf("Col for data mat row is %d, Store row from %d to %d, The col number for the file is %d\n",alldim,ib,id, dim);
	printf("The current image number is %d\n",inum);
	fclose(fptrain);
	return 1;
}

/**
 *update the weight matrix
 *@param feature_train The positive and negative training feature
 *@param current_best_classifier The current weak classifier for selection.
 *@param imith To control the shape of the sign function
 *@param curIteration The current iteration number of strong classifier
 */
int adaBoostBinary::updataWeight(CvMat *feature_train,lda_classifier *current_best_classifier,float imith, int curIteration)
{
	double min_error=current_best_classifier->minerror;
	int sample_num= pos_sample_num + neg_sample_num;
	int i=curIteration;
	//printf("the error rate in current iteration is %f\n",min_error);
	//printf("the threshold for current iteration is %e\n", current_best_classifier->pro_thresh);
	if(min_error<=0.0)
	{
		printf("the min error for current iteration is 0, so the program have to stop and break, sorry!!\n");
		return -1;
	}
	current_best_classifier->alpha=0.5*log((1.0+min_error)/(1-min_error));
	strongclassifier.push_back(*current_best_classifier);
	///update the weight
	double ztsumpos=0.0,ztsumneg=0.0;
	float error=0.0;
	float tmpscore;
	//int best_position=current_best_classifier.x*size+current_best_classifier.y;
	int binnum=current_best_classifier->bindex;
	float threshold=current_best_classifier->pro_thresh;
	int thr_sign=current_best_classifier->thr_sign;
	int colsize=feature_train->cols;

	for(int t=0; t<sample_num; t++)
	{
		//if(best_res_pos[t]==false)
			tmpscore=feature_train->data.fl[t*colsize+binnum]*thr_sign-threshold;
			double rtx=tmpscore/sqrt(tmpscore*tmpscore+imith*imith);
		    error+=cvmGet(weight,t,0)*rtx*labelsign[t];
			double tmp=cvmGet(weight,t,0)*exp(-current_best_classifier->alpha*labelsign[t]*iter_res[i*sample_num+t]);
			cvmSet(weight,t,0,tmp);
			if(labelsign[t]==1) {
				ztsumpos+=cvmGet(weight,t,0);
				//printf("*%f %f",rtx, iter_res[i*sample_num+t]);
			}
			else {
				ztsumneg+=cvmGet(weight,t,0);
				//printf("%f ",rtx);
			}
	}
	//printf("binnum=%d\n",binnum);
	//printf("current error just for check %f\n", error);
	//printf("ztsumpos=%f, ztsumneg=%f\n",ztsumpos, ztsumneg);
	if(ztsumpos+ztsumneg!=0)
	{
		for(int j=0;j<sample_num;j++)
		{
			cvmSet(weight,j,0,cvmGet(weight,j,0)/(ztsumpos+ztsumneg));
		}
	}
	return 1;
}
/**
 *Test the classification performance in the train data
 */
float adaBoostBinary::testStrongClassifier()
{
	int strongnum=strongclassifier.size();
	int sample_num = pos_sample_num + neg_sample_num;
	//printf("start to ada test...\n");
	float errorrate_ith_iter = 0.0f;
	int error_num_pos=0,error_num_neg=0;
	float score;
	for(int t=0; t<sample_num; t++)
	{
		score=0.0;
		for(int j=0;j<strongnum;j++)
		{
			double cur_alpha=strongclassifier[j].alpha*iter_res[j*sample_num+t];
			score+=cur_alpha;
		}
		if(score<=0&&labelsign[t]>0)
			error_num_pos++;
		if(score>0&&labelsign[t]<=0)
			error_num_neg++;
	}
	printf("Error number for positive %d, for negative %d\n", error_num_pos, error_num_neg);
	errorrate_ith_iter=(double)(error_num_pos+error_num_neg)/(pos_sample_num+neg_sample_num);
	printf("The test error rate is %.4f in training data\n",errorrate_ith_iter);
	//printf("finished ada test...\n\n");
	return errorrate_ith_iter;

}

}
