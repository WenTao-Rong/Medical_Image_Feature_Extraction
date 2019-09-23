# -Medical-Image-Feature-Extraction-for-Ultrasound-Breast-Image-
 Ultrasound Breast Image Feature Extraction by OpenCV 3.4.5 and C++. The code extracts the shape,texture, echo pattern, margin, posterior features.
 #Getting started
1.运行main.cpp
2.比对结果,我的结果在./data/2.jpg在我电脑提取的特征值.txt 和./data/2.jpg在我电脑的预测结果.png
3.如果结果一致,根据情况注释以下代码
*将busfeature.h busfeature.cpp里的void printAllFeature(string printpath);
*看情况注释掉busfeature.h 两个构造函数BUSFeature::BUSFeature(const char *imgpath,const contourpoint * plist,const int &plength)和 BUSFeature::BUSFeature(uchar* data,int width,int height,int format,const contourpoint *plist,const int & plength)
中的以下代码
   "
    //如果输入的图片是有显示设备信息的边框，需要通过以下代码去除，如果没有边框就注释掉
    Mat imggraytemp,imt;
    cvtColor(rgbimg,imggraytemp,CV_BGR2GRAY);
    threshold(imggraytemp,imt,10,255, THRESH_BINARY);
    Rect bbox=getLargestComponentByBinimage(imt);
    Mat rgbimg = rgbimg(bbox);
   "
