/**
 *basefunc.cpp
 *Purpose: Some usefull basic function
 *@author Shizhong Han
 *@version 1.0 03/10/2014
 */
#include<stdio.h>
#include<stdlib.h>
#include "caffe/util/basefunc.hpp"
#define CHARSIZE 6550000
/**
 *Get the line number of the input document filename
 *@param filename The input file name
 *@return The line number for the input filename
 */
namespace caffe {
int GetNumline(const char *filename)
{
    FILE *fp_feature;
    fp_feature=fopen(filename,"r");
    if (NULL==fp_feature)
    {
        printf("open file %s failed \n", filename);
        return -1;
    }
    
    char line[CHARSIZE];
    int numline=0;
    while(fgets(line, sizeof(line), fp_feature)!=NULL)
        numline++;

    printf("The line num of %s is %d\n",filename,numline);
    fclose(fp_feature);
    fp_feature = NULL;

    return numline;
}
/**
 *compare function when rank numbers from minimal to maximum
 *@param a b The input number a and b
 *@return 1 if a>b.
 */
int mintomax(const void * a, const void * b)
{
  float tmp=((scoreindex*)a)->score - ((scoreindex*)b)->score;
  if(tmp>0) return 1;
  else if(tmp==0) return 0;
  else return -1;
}
/**
 *Compare function when ranking numbers from maximal to minimal
 *@param a b The input number a and b
 *@return 1 if a<b
 */
int maxtomin(const void * a, const void * b)
{
  float tmp=((scoreindex*)b)->score - ((scoreindex*)a)->score ;
  if(tmp>0) return 1;
  else if(tmp==0) return 0;
  else return -1;
}
/**
 *write the float array of score to a file called scorefile.
 *@param scorefile The file name of score to write.
 *@param score The number array score
 *@sample_num The length of numbers in array score
 *@return 1 if it is written successfully
 */
int writefile(char *scorefile, float *score, int sample_num)
{
    FILE *fw_score=fopen(scorefile, "w");
    if(fw_score == NULL)
    {
	printf("failed to create the score file %s\n", scorefile);
        return 0;
    }
    for(int i=0; i<sample_num; i++)
    {
	fprintf(fw_score, "%f\n", score[i]);
    }
    fclose(fw_score);
    return 1;
}


/**
 *read the integer array of numbers from a file called scorefile.
 *@param scorefile The file name of score to write.
 *@param score The number array score
 *@sample_num The length of numbers in array score
 *@return 1 if writing successfully
 */
int readfile(char *scorefile, int *score, int sample_num)
{
    FILE *fw_score=fopen(scorefile, "r");
    if(fw_score == NULL)
    {
	printf("failed to create the score file %s\n", scorefile);
        return 0;
    }
    for(int i=0; i<sample_num; i++)
    {
	fscanf(fw_score, "%d\n", &score[i]);
    }
    fclose(fw_score);
    return 1;
}
/**
 *calculate the mean value of the input vector T with size n
 *@param arr the input vector
 *@param n the size of input vector
 */
template <typename T>
double Mean(const T arr[], size_t n)
{
	double mean = 0.0;
	for (size_t idx = 0; idx < n; idx++)
	{
		mean += arr[idx];
	}
	mean /= static_cast<double>(n);
	return mean;
}
template double Mean(const double arr[], size_t n);
template double Mean(const float arr[], size_t n);
/**
 *calculate the std value of the input vector T with size n
 *@param arr the input vector
 *@param n the size of input vector
 */
template <typename T>
double StDev(const T arr[], size_t n)
{
	double mean = Mean(arr, n);
	double variance = 0.0;
	for (size_t idx = 0; idx < n; idx++)
	{
		double temp = arr[idx] - mean;
		variance += temp*temp;
	}

	// Compute sample variance using Bessel's correction (see http://en.wikipedia.org/wiki/Bessel%27s_correction)
	variance /= static_cast<double>(n) - (n == 1 ? 0.0 : 1.0);

	// Standard deviation is square root of variance
	return std::sqrt(variance);
}
template double StDev(const double arr[], size_t n);
template double StDev(const float arr[], size_t n);
}

