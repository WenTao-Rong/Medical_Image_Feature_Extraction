#ifndef __RAMIMAGE__
#define __RAMIMAGE__
#include <opencv/cv.h>
using namespace cv;
#define GRAY8 CV_8UC1
#define RGB888 CV_8UC3
Mat getPicbyRAM(uchar* data,int width,int height,int format);
#endif
