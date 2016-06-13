/**
 *basefunc.h
 *Purpose: Some usefull basic function
 *@author Shizhong Han
 *@version 1.0 03/11/2014
 */
#ifndef BASEFUNC_H
#define BASEFUNC_H
#include<stdio.h>
#include <cmath>
#include<stdlib.h>
/**
 * This file is the basic function for the AdaBoost classification including the calculate the line number of a input file, compare function, write file, and calculate the mean and std value.
 */
/**
 *The struct scoreindex includes two members score and index.
 */
namespace caffe {
typedef struct tag_scoreindex
{
    float score;//actual value
    int index;//integer position
}scoreindex;
/**
 *Get the line number of the input document filename
 *@param filename The input file name
 *@return The line number for the input filename
 */
int GetNumline(const char *filename);
/**
 *compare function when rank numbers from minimal to maximum
 *@param a b The input number a and b
 *@return value 1 if a>b.
 */
int mintomax(const void * a, const void *b);
/**
 *Compare function when ranking numbers from maximum to minimal
 *@param a b The input number a and b
 *@return value 1 if a<b
 */
int maxtomin(const void *a, const void *b);
/**
 *write the float array of score to a file called scorefile.
 *@param scorefile The file name of score to write.
 *@param score The number array score
 *@sample_num The length of numbers in array score
 *@return 1 if writing successfully
 */
int writefile(char *scorefile, float *score, int sample_num);

/**
 *calculate the mean value of the input vector T with size n
 *@param arr the input vector
 *@param n the size of input vector
 */
template <typename T>
double Mean(const T arr[], size_t n);


/**
 *calculate the std value of the input vector T with size n
 *@param arr the input vector
 *@param n the size of input vector
 */
template <typename T>
double StDev(const T arr[], size_t n);

}

#endif


