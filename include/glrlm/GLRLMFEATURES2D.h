#ifndef GLRLMFEATURES2D_H_INCLUDED
#define GLRLMFEATURES2D_H_INCLUDED
#include "GLRLMFeatures.h"

template <class T>
 struct GLRLMValues
{
    T shortRunEmphasis;
    T longRunEmphasis;
	T lowGreyEmph;
	T highGreyEmph;
	T shortRunLow;
	T shortRunHigh;
	T longRunLowEmph;
	T longRunHighEmph;
	T greyNonUniformity;
	T greyNonUniformityNorm;
	T runLengthNonUniformity;  //check if I have to give the matrix
	T runLengthNonUniformityNorm;
	T greyLevelVar;
	T runLengthVar;
	T runEntropy;
	T runPercentage;
};
template <class T>
class GLRLMFEATURES2D : GLRLMFeatures<T> {
private:
	GLRLMFeatures<T> glrlm;
	double totalSum;
	typedef Mat glrlmMat;
	int directionX;
	int directionY;
	int maxRunLength;
	Mat  createGLRLMatrix2D(const Mat & inputMatrix, int ang);
	void fill2DMatrices2D(const Mat & inputMatrix, Mat  &glrlMatrix, int ang);
public:

	GLRLMFEATURES2D() {
	}
	~GLRLMFEATURES2D() {
	}
	void extractGLRLMData2D(GLRLMValues<T> &, GLRLMFEATURES2D<T> glrlmFeatures);
	void calculateAllGLRLMFEATURES2D(GLRLMFEATURES2D<T>& glrlmFeatures, const Mat&  inputMatrix, vector<T> diffFGrey);

};


template <class T>
Mat GLRLMFEATURES2D<T>::createGLRLMatrix2D(const Mat &inputMatrix, int ang) {
    int sizeMatrix = this->diffGreyLevels.size();
   // cout<<sizeMatrix<<endl;
  //不要忘了初始化
	glrlmMat GLRLMatrix(sizeMatrix,maxRunLength,CV_32FC1,cv::Scalar::all(0));


    fill2DMatrices2D(inputMatrix, GLRLMatrix,ang);
    return GLRLMatrix;
}


template <class T>
void GLRLMFEATURES2D<T>::fill2DMatrices2D(const Mat& inputMatrix, Mat  &glrlMatrix, int ang) {
    Mat inputMatrix1(inputMatrix.size(),CV_32FC1);
   inputMatrix1=inputMatrix.clone();
	T actGreyLevel = 0;
	T actElement = 0;
	int runLength = 0;
	int maxRowNr = inputMatrix1.rows;
	int maxColNr = inputMatrix1.cols;
	glrlm.getXYDirections(directionX, directionY,ang);
	//have a look at the image-matrix slide by slide (2D)
	//look for every grey level separately in every image slide
	int actGreyIndex;
	//get the grey level we are interested at the moment
	for (int row = 0; row<maxRowNr; row++) {
		for (int column = 0; column<maxColNr; column++) {
			//         //at the beginning the run length =0
			runLength = 0;
			//get the actual matrix element
			actElement = inputMatrix1.at<float>(maxRowNr - row - 1,column);
			//cout<<actElement<<endl;
			actGreyIndex = glrlm.findIndex(this->diffGreyLevels, this->diffGreyLevels.size(), actElement);
			//cout<<actGreyIndex<<endl;
			//if the actual matrix element is the same as the actual gre level
			if (!std::isnan(actElement)) {
             //   cout<<row<<"  "<<column<<actElement<<endl;
				//set the run length to 1
				runLength = 1;
				//to avoid to take an element more than once, set the element to NAN
				inputMatrix1.at<float>(maxRowNr - row - 1,column) = NAN;
				////          //now look at the matrix element in the actual direction (depends on the
				//angle we are interested at the moment
				int colValue = column + directionX;
				int rowValue = maxRowNr - 1 - (row + directionY);
				//now have a look at the following elements in the desired direction
				//stop as soon as we look at an element diifferent from our actual element
				while (colValue<maxColNr && rowValue>-1 && colValue>-1 && inputMatrix1.at<float>(rowValue,colValue) == actElement) {
					//for every element we find, count the runLength
					runLength += 1;
					inputMatrix1.at<float>(rowValue,colValue)  = NAN;
					//go further in the desired direction
					colValue += 1 * directionX;
					rowValue -= 1 * directionY;
				}
			}
			//as soon as we cannot find an element in the desired direction, count one up in the desired
			//position of the glrl-matrix
			if (actGreyIndex > -1 && runLength > 0 && runLength < glrlMatrix.cols + 1) {
                   // if ( runLength > 0 && runLength < glrlMatrix.cols + 1) {
				glrlMatrix.at<float>(actGreyIndex,runLength - 1) += 1;
			}

		}
	}
}

template <class T>
void GLRLMFEATURES2D<T>::calculateAllGLRLMFEATURES2D(GLRLMFEATURES2D<T> &glrlmFeatures, const Mat &inputMatrix, vector<T> diffGrey) {
	this->diffGreyLevels = diffGrey;
	glrlmFeatures.setEmphasisValues(0,1,1);
	T sumShortRunEmphasis = 0;
	T sumLongRunEmphasis = 0;
	T sumLowGreyEmph = 0;
	T sumHighGreyEmph = 0;
	T sumShortRunLow = 0;
	T sumShortRunHigh = 0;
	T sumLongRunLowEmph = 0;
	T sumLongRunHighEmph = 0;
	T sumGreyNonUniformity = 0;
	T sumGreyNonUniformityNorm = 0;
	T sumRunLengthNonUniformity = 0;
	T sumRunLengthNonUniformityNorm = 0;
	T sumRunPercentage = 0;
	T sumGreyLevelVar = 0;
	T sumRunLengthVar = 0;
	T sumRunEntropy = 0;

	vector<double> rowSums;
	vector<double> colSums;



	double meanGrey;
	double meanRun;

	maxRunLength = glrlm.getMaxRunLength(inputMatrix);
	int ang;
	for (int i = 0; i < 4; i++) {
		ang = 180 - i * 45;
		Mat  glrlMatrix = createGLRLMatrix2D(inputMatrix, ang);
		totalSum = glrlmFeatures.calculateTotalSum(glrlMatrix);
		rowSums = glrlmFeatures.calculateRowSums(glrlMatrix);
		colSums = glrlmFeatures.calculateColSums(glrlMatrix);
		Mat  probMatrix = glrlmFeatures.calculateProbMatrix(glrlMatrix, totalSum);

		meanGrey = glrlmFeatures.calculateMeanProbGrey(probMatrix);
		meanRun = glrlmFeatures.calculateMeanProbRun(probMatrix);
		glrlmFeatures.calculateShortRunEmphasis(rowSums, totalSum);
		sumShortRunEmphasis += this->shortRunEmphasis;
		glrlmFeatures.calculateLongRunEmphasis(rowSums, totalSum);
		sumLongRunEmphasis += this->longRunEmphasis;

		glrlmFeatures.calculateLowGreyEmph(colSums, totalSum);
		sumLowGreyEmph += this->lowGreyEmph;

		glrlmFeatures.calculateHighGreyEmph(colSums, totalSum);
		sumHighGreyEmph += this->highGreyEmph;
		glrlmFeatures.calculateShortRunLow(glrlMatrix, totalSum);
		sumShortRunLow += this->shortRunLow;
		glrlmFeatures.calculateShortRunHigh(glrlMatrix, totalSum);
		sumShortRunHigh += this->shortRunHigh;
		glrlmFeatures.calculateLongRunLowEmph(glrlMatrix, totalSum);
		sumLongRunLowEmph += this->longRunLowEmph;
		glrlmFeatures.calculateLongRunHighEmph(glrlMatrix, totalSum);
		sumLongRunHighEmph += this->longRunHighEmph;
		glrlmFeatures.calculateGreyNonUniformity(colSums, totalSum);
		sumGreyNonUniformity += this->greyNonUniformity;
		glrlmFeatures.calculateGreyNonUniformityNorm(colSums, totalSum);
		sumGreyNonUniformityNorm += this->greyNonUniformityNorm;
		glrlmFeatures.calculateRunLengthNonUniformity(rowSums, totalSum);
		sumRunLengthNonUniformity += this->runLengthNonUniformity;
		glrlmFeatures.calculateRunLengthNonUniformityNorm(rowSums, totalSum);
		sumRunLengthNonUniformityNorm += this->runLengthNonUniformityNorm;
		glrlmFeatures.calculateGreyLevelVar(probMatrix, meanGrey);
		sumGreyLevelVar += this->greyLevelVar;
        glrlmFeatures.calculateRunPercentage(inputMatrix,totalSum, 4);
		sumRunPercentage += this->runPercentage;

		glrlmFeatures.calculateRunLengthVar(probMatrix, meanRun);
		sumRunLengthVar += this->runLengthVar;

		glrlmFeatures.calculateRunEntropy(probMatrix);
		sumRunEntropy += this->runEntropy;

	}

	this->shortRunEmphasis = sumShortRunEmphasis / 4;
	this->longRunEmphasis = sumLongRunEmphasis /  4;
	this->lowGreyEmph = sumLowGreyEmph / 4;
	this->highGreyEmph = sumHighGreyEmph / 4;
	this->shortRunLow = sumShortRunLow / 4;
	this->shortRunHigh = sumShortRunHigh / 4;
	this->longRunLowEmph = sumLongRunLowEmph / 4;
	this->longRunHighEmph = sumLongRunHighEmph / 4;
	this->greyNonUniformity = sumGreyNonUniformity / 4;
	this->greyNonUniformityNorm = sumGreyNonUniformityNorm / 4;
	this->runLengthNonUniformity = sumRunLengthNonUniformity / 4;
	this->runLengthNonUniformityNorm = sumRunLengthNonUniformityNorm / 4;
	this->runPercentage = sumRunPercentage / 4;

	this->greyLevelVar = sumGreyLevelVar / 4;
	this->runLengthVar = sumRunLengthVar / 4;
	this->runEntropy = sumRunEntropy / 4;
	this->runPercentage = sumRunPercentage;
}



template <class T>
void GLRLMFEATURES2D<T>::extractGLRLMData2D(GLRLMValues<T> & glrlmData, GLRLMFEATURES2D<T> glrlmFeatures) {

	glrlmData.shortRunEmphasis=glrlmFeatures.shortRunEmphasis;
	glrlmData.longRunEmphasis=glrlmFeatures.longRunEmphasis;
	glrlmData.lowGreyEmph=glrlmFeatures.lowGreyEmph;
	glrlmData.highGreyEmph=glrlmFeatures.highGreyEmph;
	glrlmData.shortRunLow=glrlmFeatures.shortRunLow;
	glrlmData.shortRunHigh=glrlmFeatures.shortRunHigh;
	glrlmData.longRunLowEmph=glrlmFeatures.longRunLowEmph;
	glrlmData.longRunHighEmph=glrlmFeatures.longRunHighEmph;
	glrlmData.greyNonUniformity=glrlmFeatures.greyNonUniformity;
	glrlmData.greyNonUniformityNorm=glrlmFeatures.greyNonUniformityNorm;
	glrlmData.runLengthNonUniformity=glrlmFeatures.runLengthNonUniformity;   //check if I have to give the matrix
	glrlmData.runLengthNonUniformityNorm=glrlmFeatures.runLengthNonUniformityNorm;
	glrlmData.greyLevelVar=glrlmFeatures.greyLevelVar;
	glrlmData.runLengthVar=glrlmFeatures.runLengthVar;
	glrlmData.runEntropy=glrlmFeatures.runEntropy;
	glrlmData.runPercentage=glrlmFeatures.runPercentage;
}

#endif // GLRLMFEATURES2D_H_INCLUDED

