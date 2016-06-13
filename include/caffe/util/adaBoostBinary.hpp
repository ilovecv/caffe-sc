/**
 *The AdaBoost algorithm file
 *@author Shizhong Han
 *@verion 1.0 03/11/2014
 */
#ifndef ADABOOSTBINARY_H
#define ADABOOSTBINARY_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "cxcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <numeric>
#include <omp.h>
#include <algorithm>
#include "caffe/util/basefunc.hpp"
#define SIZE 200
using namespace std;
using namespace cv;
/**
 *Binary tree weak classifier for AdaBoost
 */
namespace caffe {

/**
 *Binary tree weak classifier for AdaBoost
 */
typedef struct tag_binaryTreeClassifier
{
    int x;//x position in face image
    int y;//y position in face image
    int bindex;//The index for current selected weak classifier
    double alpha;//The alpha value is the weight of the AdaBoost strong classifier
    double pro_thresh;//The threshold for the binary tree weak classifier
    int thr_sign;//The sign for the current weak classifier. The sign can be used to keep the mean of positive data is always bigger than the negative data.
    double minerror;//The error in current iteration
    double real_margin;//The constrain parameter of margin for the weak classifier
}lda_classifier;
/**
 * The AdaBoost binary classifier including reading the training data, selecting the weak classifier, and making prediction in the training data
 */
class adaBoostBinary
{
private:
	CvMat *weight;//The weight for each training samples
	float *iter_res;//The weak classifier classification results for each iteration and each training image
    int *labelsign;//The binary ground truth label -1 or 1 for each training image
	int maxIterationNum;//The maximum training iteration
	int pos_sample_num;//The number of positive training sample
	int neg_sample_num;//The number of negative training sample
	double rate_margin;//The margin rate parameter to control the real margin


public:
	 vector<lda_classifier> strongclassifier;
	 adaBoostBinary();
     virtual ~adaBoostBinary();
    /**
     *call the memory for the AdaBoost variable
     *@param posnum The positive sample number
     *@param negnum The negative sample number
     *@param iterNum The maximum iteration number
     *@return 1 If it is successful
     */

    int allocateMemory(int posnum, int negnum, int iterNum);
     /**
      *Initialize the training label and transfer the label from 0 1 to short -1 and 1
     *@param label The training label of 0 and 1 value
     *@return 1 If it is successful
     */
    int Initialize(CvMat *label);
    /**
    *Initialize weight matrix using the summation value of sum
    *@param label The training label of 0 and 1 value
    *@return 1 If it is successful
    */
    int InitializeWeightFile(CvMat *label);
    /**
     *Choose the weak classifier using the fast method just based on the distribution of the weight
     *@param data The positive and negative training samples
     *@param label the ground truth label of the training samples
     *@param current_best_classifier The current weak classifier for selection.
     *@param bindexlabel The BOOL vector to ensure that each feature can only be choose once
     *@param imith To control the shape of the sign function
     *@param curIteration The current iteration number of strong classifier
     */
    int choosefeature_fast(CvMat *data,CvMat *label, lda_classifier *current_best_classifier, bool *bindexlabel, float imith, int curIteration);
    /**
     *Choose the weak classifier selection process for the Convolutional Neural Network
     *@param data The positive and negative training samples
     *@param label the ground truth label of the training samples
     *@param current_best_classifier The current weak classifier for selection.
     *@param bindexlabel The BOOL vector to ensure that each feature can only be choose once
     *@param sign The history sign for each weak classifier 0 1 -1. 0 means that it's selected in the history. 1 means mean value of positive is bigger than negative
     *@param thresh The history threshold for each weak classifier for Convolutional Neural Network
     *@param imith To control the shape of the sign function
     *@param curIteration The current iteration number of strong classifier
     */
    template <typename Dtype>
    int choosefeature_fastupdate(CvMat *data,CvMat *label, lda_classifier *current_best_classifier, bool* bindexlabel,Dtype *sign,Dtype *thresh, float imith, int curIteration);
    /**
     *Load the feature continually
     *@param feature_train The positive and negative training feature
     *@param trainfile The file path for reading the train feature
     *@param ib The beginning row in the feature_train for current data reading from trainfile. It's used when the data is saved in multiple files.
     *@param id The ending row in the feature_train for current data reading from trainfile
     *@param dim is the dimension of the training feature
     *@param inum The current set number if the data is divided into multiple set and saved in multiple files
     */
    int LoadDataAllFeat_set(CvMat *feature_train, const char* trainfile,int ib, int id,int dim,int &inum);
    
    /**
     *update the weight matrix
     *@param feature_train The positive and negative training feature
     *@param current_best_classifier The current weak classifier for selection.
     *@param imith To control the shape of the sign function
     *@param curIteration The current iteration number of strong classifier
     */
    int updataWeight(CvMat *feature_train,lda_classifier *current_best_classifier, float imith, int curIteration);
    /**
     *Test the classification performance in the train data
     */
    float testStrongClassifier();



};


}

#endif


