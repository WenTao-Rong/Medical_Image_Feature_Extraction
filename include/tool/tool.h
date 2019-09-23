#ifndef _TOOL_
#define _TOOL_
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "opencv/highgui.h" //include it to use GUI functions.
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../blob/blob.h"
#include "../blob/BlobResult.h"
#include "../basic/contourpoint.h"
#include <opencv/cv.h>

using namespace cv;
using namespace std;
float graysdiff(const Mat &labelgray, const Mat & img_region,int t_len,bool entiretumor);
float grayavgsep(const Mat &labelgray,const Mat & img_region,int t_len);
t_PointList getLargestComponentPoint(const Mat binary);
Rect getLargestComponentByBinimage( const Mat binary);
Mat getContourByRedline(const Mat& src);
void gdfffunderlr( Mat &labelgray,Mat &tumorUnder,Mat& tumorUnderLR,Range& r32,int underheight);
#endif

