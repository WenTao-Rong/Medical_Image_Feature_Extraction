#ifndef GLRLMFEATURES_H_INCLUDED
#define GLRLMFEATURES_H_INCLUDED

#include <iostream>
#include <algorithm>
#include <vector>
#include "cv.h"
#include "helpFunctions.h"
#include "highgui.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
template <class T>
class GLRLMFeatures  {
    private:
        Mat mat;
        double totalSum;
        void extractGLRLMData(vector<T> &glrlmData, GLRLMFeatures<T> glrlmFeatures);
        int totalNrVoxels;
        //store different grey levels in vector
    public:

        int maxRunLength;
        float powRow;
        float powCol;
		int calculateExtEmph;
        void defineGLRLMFeatures(vector<string> &features);

        void getXYDirections(int &directionX, int &directionY, int angle);

        vector<T> diffGreyLevels;
        GLRLMFeatures(){
        }

        double shortRunEmphasis;
        double longRunEmphasis;
        double lowGreyEmph;
        double highGreyEmph;
        double shortRunLow;
        double shortRunHigh;
        double longRunLowEmph;
        double longRunHighEmph;
        double greyNonUniformity;
        double greyNonUniformityNorm;
        double runLengthNonUniformity;
        double runLengthNonUniformityNorm;
        double runPercentage;
        double greyLevelVar;
        double runLengthVar;
        double runEntropy;

        vector<double> calculateRowSums(const Mat& glrlmatrix);
        vector<double> calculateColSums(const Mat & glrlmatrix);
		int findIndex(vector<T> array, int size, T target);

		void setEmphasisValues(int extEmph=0, double powRow=1, double powCol=1);
        double calculateTotalSum(const Mat& glrlMatrix);
        int getMaxRunLength(const Mat & inputMatrix);

        void calculateShortRunEmphasis(vector<double> colSums, double totalSum);
        void calculateLongRunEmphasis(vector<double> colSums, double totalSum);
        void calculateLowGreyEmph(vector<double> colSums, double totalSum);
        void calculateHighGreyEmph(vector<double> colSums, double totalSum);
        void calculateShortRunLow(const Mat& glrlmatrix, double totalSum);

        void calculateShortRunHigh(const Mat& glrlmatrix, double totalSum);
         int calculateTotalNrVoxels(const Mat& inputMatrix);
        void calculateRunPercentage(const Mat& inputMatrix, double totalSum, int nrNeighbor);
        void calculateLongRunLowEmph(const Mat& glrlmatrix, double totalSum);
        void calculateLongRunHighEmph(const Mat& glrlmatrix, double totalSum);
        void calculateGreyNonUniformity(vector<double> colSums, double totalSum);
        void calculateGreyNonUniformityNorm(vector<double> colSums, double totalSum);
        void calculateRunLengthNonUniformityNorm(vector<double> rowSums, double totalSum);
        void calculateRunLengthNonUniformity(vector<double> rowSums, double totalSum);


        Mat calculateProbMatrix(const Mat &glrlmatrix, double totalSum);
        double calculateMeanProbGrey(const Mat & probMatrix);
        void calculateGreyLevelVar(const Mat &probMatrix, double mean);
        double calculateMeanProbRun(const Mat &probMatrix);
        void calculateRunLengthVar(const Mat& probMatrix, double meanRun);
        void calculateRunEntropy(const Mat & probMatrix);

};
template <class T>
int GLRLMFeatures<T>::findIndex(vector<T> array, int size, T target) {
	int i = 0;
	while ((i < size) && (array[i] != target)) i++;
	return (i < size) ? (i) : (-1);
}

//set the exponential values to the user specified values
//this is part of the novel and uncommon feature section
template<class T>
void GLRLMFeatures<T>::setEmphasisValues(int extEmph, double row, double col) {
	calculateExtEmph = extEmph;
	powRow = row;
	powCol = col;
}

//from the GLRL-Matrix calculate the probability matrx
//do this by dividing every matrix elemnt with the total nr. of voxels
template <class T>
Mat GLRLMFeatures<T>::calculateProbMatrix(const Mat &glrlmatrix, double totalSum){
    Mat probMatrix(glrlmatrix.size(),CV_32FC1);
	 glrlmatrix.convertTo(probMatrix,CV_32FC1);
     MatIterator_<float> itbegin, itend;
	for(itbegin = probMatrix.begin<float>(), itend = probMatrix.end<float>(); itbegin != itend; itbegin++)
    {
        *itbegin /=totalSum ;
    }
    return probMatrix;
}

/*!
\brief calculateMeanProbGrey
@param Mat probMatrix matrix filled with the probabilities

calculates the mean probability of the appearance of every grey level
TODO change bordwers in for loop (probMatrix.shape())
*/
template <class T>
double GLRLMFeatures<T>::calculateMeanProbGrey( const Mat& probMatrix){
    double mean=0;
    for(int i=0; i<probMatrix.rows; i++){
      const float* Data = probMatrix.ptr<const float>(i);
        for(int j=0; j<probMatrix.cols; j++){
			if (!std::isnan(diffGreyLevels[i] *Data[j])) {
				mean += diffGreyLevels[i] *Data[j];
			}
			else {
				mean += 0;
			}
        }
    }
    return mean;
}

//calcuöate the mean probability of the runlength
template <class T>
double GLRLMFeatures<T>::calculateMeanProbRun(const Mat & probMatrix){
    double mean = 0;
    for(int i = 0; i < probMatrix.rows; i++){
           const float* Data = probMatrix.ptr<const float>(i);
        for(int j = 0; j < probMatrix.cols; j++){
			if (!std::isnan(j*Data[j])) {
				mean += j*Data[j];
			}
			else {
				mean += 0;
			}
        }
    }
    return mean;
}

/*!
\brief getMaxRunLength
The maximal run length is the maximal size of one dimension
*/
template <class T>
int GLRLMFeatures<T>::getMaxRunLength(const Mat & inputMatrix){
    maxRunLength = std::max(inputMatrix.rows, inputMatrix.cols);
    return maxRunLength;
}



/*!
calculate the sum of all matrix elements
*/
template<class T>
double GLRLMFeatures<T>::calculateTotalSum(const Mat& glrlmatrix){

//    double sum =0;
//    for(int row = 0; row < glrlmatrix.rows; row++){
//      const float* Data = glrlmatrix.ptr<const float>(row);
//        for(int col = 0; col < glrlmatrix.cols; col++){
//			sum = sum+ Data[col];
//        }
//}

  double sum = cv::sum(glrlmatrix)[0];
return sum;
}

/*!
\brief calculateRowSums
@param Mat glrlmatrix : GLRM matrix

calculates the sum of rows and stores them in the vector rowSums
*/
template<class T>
vector<double> GLRLMFeatures<T>::calculateRowSums(const Mat& glrlmatrix){
  vector<double> rowSums;
    rowSums.clear();
    double sum =0;
    for(int col = 0; col < glrlmatrix.cols; col++){
        sum = 0;
        for(int row = 0; row < glrlmatrix.rows; row++){
			sum = sum+ glrlmatrix.at<const float>(row,col);
        }
        //cout<<"sum="<<sum<<endl;
        rowSums.push_back(sum);

    }
    return rowSums;
}


/*!

\brief calculateColSums
@param Mat glrlmatrix : GLRM matrix

calculates the sum of columns and stores them in the vector colSums
*/
template<class T>
vector<double> GLRLMFeatures<T>::calculateColSums(const Mat & glrlmatrix){
    int sum = 0;
    vector<double> colSums;
    colSums.clear();
    for(int row=0; row<glrlmatrix.rows; row++){
        sum =0;
       const float* Data = glrlmatrix.ptr<const float>(row);
        for(int col=0; col<glrlmatrix.cols; col++){
            sum += Data[col];
            }
        colSums.push_back(sum);
    }
    return colSums;
}

/*!
\brief getXYDirections
@param int directionX
@param int directionY
@param int angle

The function gets directionX and directionY as reference. Depending on the angle value,
the parameter are set: \n
angle == 180 :  go one pixel/voxel in x-direction; no move in y-direction \n
angle == 90 :  no move in x-direction; go one pixel/voxel in y-direction \n
angle == 45 : go one pixel/voxel in x-direction; go one pixel/voxel in y direction \n
angle == 135 : go minus one pixel/voxel in x-direction; one pixel/voxel in y direction
*/
template <class T>
void GLRLMFeatures<T>::getXYDirections(int &directionX, int &directionY, int angle){
    if(angle==0){
        directionX=0;
        directionY=0;

    }
        //if angle is 180 degrees, only look in x direction
    if(angle==180){
        directionX=1;
        directionY=0;
    }
    //if angle is 90 degrees only look in y direction
    else if(angle==90){
        directionX=0;
        directionY=1;
    }
    //if angle is in 45 degrees direction look in x and y direction
    else if(angle==45){
        directionX=1;
        directionY=1;
    }
    else if(angle==135){
        directionX=-1;
        directionY=1;
    }
    else{
        std::cout<<"Incorrect angle!"<<std::endl;
    }
}

/*!
\brief calculateTotalNrVoxels
@param Mat glrlmatrix : GLRLM matrix

calculates the total number of voxels of one slice of the matrix

*/
template <class T>
int GLRLMFeatures<T>::calculateTotalNrVoxels(const Mat & inputMatrix){
    vector<T> vectorSliceElements;
    for(int row = 0; row < inputMatrix.rows; row++){
             const float* Data = inputMatrix.ptr<const float>(row);
        for(int col = 0; col < inputMatrix.cols; col++){
            if(!std::isnan(Data[col])){
                vectorSliceElements.push_back(Data[col]);
            }
        }
    }
    totalNrVoxels=vectorSliceElements.size();
	return totalNrVoxels;
}
/*!
\brief calculateShortRunEmphasis
@param vector<double> rowSums : vector of the sums of the rows
@param double totalSum : sum of all matrix elements

This feature emphasizes the short runs. The higher the value, the more short runs are in the matrix.
*/

template<class T>
void GLRLMFeatures<T>::calculateShortRunEmphasis(vector<double> rowSums, double totalSum){
    shortRunEmphasis = 0;
	if (totalSum != 0) {
		for(int j=0; j<rowSums.size(); j++){
			if (calculateExtEmph == 0) {
				if (!std::isnan(rowSums[j] / pow(j + 1, 2)) ){

					shortRunEmphasis += rowSums[j] / pow(j + 1, 2);
				}
				else {
					shortRunEmphasis += 0;
				}
			}
			else {
				if (!std::isnan(rowSums[j] / pow(j + 1, powRow))) {
					shortRunEmphasis += rowSums[j] / pow(j + 1, powRow);
				}
				else {
					shortRunEmphasis += 0;
				}
			}
		}

		shortRunEmphasis = shortRunEmphasis / totalSum;
	}
}


/*!
\brief calculateLongRunEmphasis
@param vector<double> rowSums : vector of the sums of the rows
@param double totalSum : sum of all matrix elements

This feature emphasizes the long runs. The higher the value, the more long runs are in the matrix.
*/

template<class T>
void GLRLMFeatures<T>::calculateLongRunEmphasis(vector<double> rowSums, double totalSum){
    longRunEmphasis=0;
	if (totalSum != 0) {
		for(int j=0; j<rowSums.size(); j++){
			if (calculateExtEmph == 0) {
				if (!std::isnan(rowSums[j] * pow(j + 1, 2))) {
					longRunEmphasis += rowSums[j] * pow(j + 1, 2);
				}
				else {
					longRunEmphasis += 0;
				}
			}
			else {
				if (!std::isnan(rowSums[j] * pow(j + 1, powRow))) {
					longRunEmphasis += rowSums[j] * pow(j + 1, powRow);
				}
			}
		}

		longRunEmphasis = longRunEmphasis / totalSum;
	}
}


/*!
\brief calculateLowGreyEmph
@param vector<double> colSums : vector of the sums of the columns
@param double totalSum : sum of all matrix elements

This feature emphasizes the low grey levels. The higher the value, the more low grey levels are in the matrix.
*/
template<class T>
void GLRLMFeatures<T>::calculateLowGreyEmph(vector<double> colSums, double totalSum){
    lowGreyEmph=0;
	if (totalSum != 0) {
		for(int i=0; i<colSums.size(); i++){
			if (calculateExtEmph == 0) {
				if (diffGreyLevels[i] != 0) {
					lowGreyEmph += colSums[i] / pow(diffGreyLevels[i], 2);
				}
			}
			else {
				if (diffGreyLevels[i] != 0) {
					lowGreyEmph += colSums[i] / pow(diffGreyLevels[i], powCol);
				}
			}
		}
		lowGreyEmph = lowGreyEmph / totalSum;
	}
}


/*!
\brief calculateHighGreyEmph
@param vector<double> colSums : vector of the sums of the columns
@param double totalSum : sum of all matrix elements

This feature emphasizes the high grey levels. The higher the value, the more high grey levels are in the matrix.
*/
template <class T>
void GLRLMFeatures<T>::calculateHighGreyEmph(vector<double> colSums, double totalSum){
    highGreyEmph=0;
	if (totalSum != 0) {
		for(int i=0; i<colSums.size(); i++){
			if (calculateExtEmph == 0) {
				if (!std::isnan(colSums[i] * pow(diffGreyLevels[i], 2))) {
					highGreyEmph += colSums[i] * pow(diffGreyLevels[i], 2);
				}

			}
			else {
				if (!std::isnan(colSums[i] * pow(diffGreyLevels[i], powCol))) {
					highGreyEmph += colSums[i] * pow(diffGreyLevels[i], powCol);
				}

			}
		}

		highGreyEmph = highGreyEmph / totalSum;
	}
}


/*!
\brief calculateShortRunLow
@param Mat glrlmatrix : GLCM matrix
@param double totalSum : sum of all matrix elements

This feature emphasizes the low grey levels which habe a short run. The higher the value, the more low grey levels with short runs are in the matrix.
*/
template <class T>
void GLRLMFeatures<T>::calculateShortRunLow(const Mat &glrlmatrix, double totalSum){
    shortRunLow = 0;
	if (totalSum != 0) {
		for(int row = 0; row < glrlmatrix.rows; row++){
                  const float* Data = glrlmatrix.ptr<const float>(row);
			for(int col = 1; col < glrlmatrix.cols+1; col++){
				if (calculateExtEmph == 0) {
					if (!std::isnan(Data[col - 1] / (pow(diffGreyLevels[row], 2)*pow(col, 2)))) {
						shortRunLow += Data[col - 1] / (pow(diffGreyLevels[row], 2)*pow(col, 2));
					}
					else {
						shortRunLow += 0;
					}
				}
				else {
					if (!std::isnan(Data[col - 1] / (pow(diffGreyLevels[row], 2)*pow(col, powCol)))) {
						shortRunLow += Data[col - 1] / (pow(diffGreyLevels[row], powRow)*pow(col, powCol));
					}
				}
			}
		}

		shortRunLow = shortRunLow / totalSum;
	}
}


/*!
\brief calculateShortRunHigh
@param Mat glrlmatrix : GLCM matrix
@param double totalSum : sum of all matrix elements

This feature emphasizes the high grey levels which habe a short run. The higher the value, the more high grey  levels with short runs are in the matrix.
*/
template <class T>
void GLRLMFeatures<T>::calculateShortRunHigh(const Mat& glrlmatrix, double totalSum){
    shortRunHigh = 0;
	if (totalSum != 0) {
		for(int row = 0; row < glrlmatrix.rows; row++){
               const  float* Data = glrlmatrix.ptr<const float>(row);
			for(int col = 1; col < glrlmatrix.cols+1; col++){
				if (calculateExtEmph == 0) {
					if (!std::isnan(pow(diffGreyLevels[row], 2)*Data[col - 1] / pow(col, 2))) {
						shortRunHigh += pow(diffGreyLevels[row], 2)*Data[col - 1] / pow(col, 2);
					}
				}
				else {
					if (!std::isnan(pow(diffGreyLevels[row], 2)*Data[col - 1] / pow(col, powCol))) {
						shortRunHigh += pow(diffGreyLevels[row], powRow)*Data[col - 1] / pow(col, powCol);
					}

				}
			}
		}

		shortRunHigh = shortRunHigh / totalSum;
	}
}

/*!
\brief calculateLongRunLowEmph
@param Mat glrlmatrix : GLCM matrix
@param double totalSum : sum of all matrix elements

This feature emphasizes the low grey levels which habe a long run. The higher the value, the more low grey levels with long runs are in the matrix.
*/
template <class T>
void GLRLMFeatures<T>::calculateLongRunLowEmph(const Mat & glrlmatrix, double totalSum){
    longRunLowEmph = 0;
	if (totalSum != 0) {
		for(int row = 0; row < glrlmatrix.rows; row++){
                 const float* Data = glrlmatrix.ptr<const float>(row);
			for(int col = 1; col < glrlmatrix.cols+1; col++){
				if(diffGreyLevels[row]!=0){
					if (calculateExtEmph == 0) {
						if (!std::isnan(pow(col, 2)*Data[col - 1] / pow(diffGreyLevels[row], 2))) {
							longRunLowEmph += pow(col, 2)*Data[col - 1] / pow(diffGreyLevels[row], 2);
						}
					}
					else {
						if (!std::isnan(pow(col, 2)*Data[col - 1] / pow(diffGreyLevels[row], powRow))) {
							longRunLowEmph += pow(col, powCol)*Data[col - 1] / pow(diffGreyLevels[row], powRow);
						}
					}
				}
			}
		}

		longRunLowEmph = longRunLowEmph / totalSum;
	}
}


/*!
\brief calculateLongRunHighEmph
@param Mat glrlmatrix : GLCM matrix
@param double totalSum : sum of all matrix elements

This feature emphasizes the high grey levels which habe a long run. The higher the value, the more high grey levels with long runs are in the matrix.
*/
template <class T>
void GLRLMFeatures<T>::calculateLongRunHighEmph(const Mat& glrlmatrix, double totalSum){
    longRunHighEmph=0;
	if (totalSum != 0) {
		for (int row = 0; row < glrlmatrix.rows; row++) {
              const  float* Data = glrlmatrix.ptr<const float>(row);
			for (int col = 1; col < glrlmatrix.cols + 1; col++) {
				if (calculateExtEmph == 0) {
					if (!std::isnan(pow(col, 2)*pow(diffGreyLevels[row], 2)*Data[col - 1])) {
						longRunHighEmph += pow(col, 2)*pow(diffGreyLevels[row], 2)*Data[col - 1];
					}
				}
				else {
					if (!std::isnan(pow(col, 2)*pow(diffGreyLevels[row], powRow)*Data[col - 1])) {
						longRunHighEmph += pow(col, powCol)*pow(diffGreyLevels[row], powRow)*Data[col - 1];
					}
				}
			}
		}
		longRunHighEmph = longRunHighEmph / totalSum;
	}
}

/*!
\brief calculateGreyNonUniformity
@param vector<double> colSums : vector of the column sums
@param double totalSum : sum of all matrix elements

This features is a measure for the distribution of the grey levels in the image matrix. \n
The more equally distrbuted the runs of the grey levels are, the lower is the value.
*/
template <class T>
void GLRLMFeatures<T>::calculateGreyNonUniformity(vector<double> colSums, double totalSum){
    greyNonUniformity = 0;
    greyNonUniformity = for_each(colSums.begin(), colSums.end(), square_accumulate<double>()).result();
	if (totalSum != 0) {
		greyNonUniformity = greyNonUniformity / totalSum;
	}
	else {
		greyNonUniformity = 0;
	}
}


/*!
\brief calculateGreyNonUniformityNorm
@param vector<double> colSums : vector of the column sums
@param double totalSum : sum of all matrix elements

This features is a normalized version of the grey-non-uniformity feature.
*/
template <class T>
void GLRLMFeatures<T>::calculateGreyNonUniformityNorm(vector<double> colSums, double totalSum){
    greyNonUniformityNorm = for_each(colSums.begin(), colSums.end(), square_accumulate<double>()).result();
	if (totalSum != 0) {
		greyNonUniformityNorm = greyNonUniformityNorm / pow(totalSum, 2);
	}
	else {
		greyNonUniformityNorm = 0;
	}

}


/*!
\brief calculateRunLengthNonUniformity
@param vector<double> colSums : vector of the column sums
@param double totalSum : sum of all matrix elements

This feature is a measurement for the distribution of the run length. \n
The lower this value is, the more equally the run length are distributed.
*/
template <class T>
void GLRLMFeatures<T>::calculateRunLengthNonUniformity(vector<double> rowSums, double totalSum){
    runLengthNonUniformity=for_each(rowSums.begin(), rowSums.end(), square_accumulate<double>()).result();
	if (totalSum != 0) {
		runLengthNonUniformity = runLengthNonUniformity / totalSum;
	}
	else {
		runLengthNonUniformity = 0;
	}
}


/*!
\brief calculateRunLengthNonUniformityNorm
@param vector<double> colSums : vector of the column sums
@param double totalSum : sum of all matrix elements

This is a normalised version of the run-length non uniformity feature.
*/
template <class T>
void GLRLMFeatures<T>::calculateRunLengthNonUniformityNorm(vector<double> rowSums, double totalSum){
    runLengthNonUniformityNorm=for_each(rowSums.begin(), rowSums.end(), square_accumulate<double>()).result();
	if (totalSum != 0) {
		runLengthNonUniformityNorm = runLengthNonUniformityNorm / pow(totalSum, 2);
	}
	else {
		runLengthNonUniformityNorm = 0;
	}
}


/*!
\brief calculateRunPercentage
@param Mat glrlmatrix : GLRLM matrix
@param double totalSum : sum of all matrix elements

calculates the fraction of runs appearing in the matrix and potential runs
*/
template <class T >
void GLRLMFeatures<T>::calculateRunPercentage(const Mat & inputMatrix, double totalSum, int nrNeighbor){
	//if (depth == 0) {
	//	totalNrVoxels = inputMatrix.shape()[0] * inputMatrix.shape()[1] * inputMatrix.shape()[2];
	//}
	//else {
		totalNrVoxels = calculateTotalNrVoxels(inputMatrix);
	//}
	if ((totalNrVoxels)*nrNeighbor != 0) {
		runPercentage = totalSum / ((totalNrVoxels)*nrNeighbor);
	}
	else {
		runPercentage = 0;
	}
}


/*!
\brief calculateGreyLevelVar
@param Mat probMatrix : probability matrix
@param double meanGrey : mean value of the grey levels

calculates the variance of grey levels \n
the lower the value, the more homogeneous is the region
*/
template <class T >
void GLRLMFeatures<T>::calculateGreyLevelVar(const Mat& probMatrix, double meanGrey){
    greyLevelVar=0;
    for(int i=0; i<probMatrix.rows; i++){
            const float* Data = probMatrix.ptr<const float>(i);
        for(int j= 0; j<probMatrix.cols; j++){
			if (!std::isnan(pow((diffGreyLevels[i] - meanGrey), 2)*Data[j])) {
				greyLevelVar += pow((diffGreyLevels[i] - meanGrey), 2)*Data[j];
			}
        }
    }
}


/*!
\brief calculateRunLengthVar
@param Mat probMatrix : probability matrix
@param double meanRun : mean value of the run length

calculates the variance of run length \n
the lower the value, the more homogeneous is the region
*/
template <class T>
void GLRLMFeatures<T>::calculateRunLengthVar(const Mat & probMatrix, double meanRun){
    runLengthVar = 0;
    for(int i=0; i<probMatrix.rows; i++){
             const float* Data = probMatrix.ptr<const float>(i);
        for(int j= 0; j<probMatrix.cols; j++){
			if (!std::isnan(pow((j - meanRun), 2)*Data[j])) {
				runLengthVar += pow((j - meanRun), 2)*Data[j];
			}
        }
    }
}


/*!
\brief calculateRunEntropy
@param Mat probMatrix : probability matrix

calculates the entropy of the probability matrix
*/
template <class T>
void GLRLMFeatures<T>::calculateRunEntropy(const Mat & probMatrix){
    runEntropy=0;
    for(int i=0; i<probMatrix.rows; i++){
      const float* Data = probMatrix.ptr<const float>(i);
        for(int j= 0; j<probMatrix.cols; j++){
            if(Data[j]>0){
				if (!std::isnan(Data[j] * log2(Data[j]))) {
					runEntropy -= Data[j] * log2(Data[j]);
				}

            }
        }
    }
}


template <class T>
void GLRLMFeatures<T>::extractGLRLMData(vector<T> &glrlmData, GLRLMFeatures<T> glrlmFeatures){

    glrlmData.push_back(glrlmFeatures.shortRunEmphasis);
    glrlmData.push_back(glrlmFeatures.longRunEmphasis);
    glrlmData.push_back(glrlmFeatures.lowGreyEmph);
    glrlmData.push_back(glrlmFeatures.highGreyEmph);
    glrlmData.push_back(glrlmFeatures.shortRunLow);
    glrlmData.push_back(glrlmFeatures.shortRunHigh);
    glrlmData.push_back(glrlmFeatures.longRunLowEmph);
    glrlmData.push_back(glrlmFeatures.longRunHighEmph);
    glrlmData.push_back(glrlmFeatures.greyNonUniformity);
    glrlmData.push_back(glrlmFeatures.greyNonUniformityNorm);
    glrlmData.push_back(glrlmFeatures.runLengthNonUniformity);   //check if I have to give the matrix
    glrlmData.push_back(glrlmFeatures.runLengthNonUniformityNorm);
    glrlmData.push_back(glrlmFeatures.runPercentage);
    glrlmData.push_back(glrlmFeatures.greyLevelVar);
    glrlmData.push_back(glrlmFeatures.runLengthVar);
    glrlmData.push_back(glrlmFeatures.runEntropy);

}


template <class T>
void GLRLMFeatures<T>::defineGLRLMFeatures(vector<string> &features){
    features.push_back("short run emphasis");
    features.push_back("long runs emphasis");
    features.push_back("Low grey level run emphasis");
    features.push_back("High grey level run emphasis");
    features.push_back("Short run low grey level emphasis");
    features.push_back("Short run high grey level emphasis");
    features.push_back("Long run low grey level emphasis");
    features.push_back("Long run high grey level emphasis");
    features.push_back("Grey level non uniformity");
    features.push_back("Grey level non uniformity normalized");
    features.push_back("Run length non uniformity");
    features.push_back("Run length non uniformity normalized");
    features.push_back("Grey level variance");
    features.push_back("Run length variance");
    features.push_back("Run entropy");

}

#endif //GLRLMFEATURES_H_INCLUDED

