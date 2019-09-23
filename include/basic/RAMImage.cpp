#include "RAMImage.h"

Mat getPicbyRAM(uchar* data,int width,int height,int format)
{
 	Mat imm(height,width,format, data);
	return imm;
}
