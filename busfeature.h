#ifndef _BUSFEATURE_
#define _BUSFEATURE_
#include "busheaders.h"
using namespace std;
using namespace cv;
class BUSFeature
{
    private:
    int imgcols,imgrows;
    //图像的宽度（列数）,图像的高度（行数）
    Mat rgbimg,grayimg,edge,label,label255;
      //rgbimg grayimg edge label label255分别代表输入的rgb图和灰度图,和经过手动或自动分割后得到的肿瘤边界图（值为1时是肿瘤边界的像素点），肿瘤的真实标签（二值图）,肿瘤的真实标签（0 255）
    vector<Point> allPoint;
     //allPoint 以Opencv Point结构保存肿瘤边界点的坐标信息
    TextureEValues glcmfeature;
    //GlCM feature of tumor save in a struct including  float energy,contrast, homogenity,entropy;
    GLRLMValues<float> glrlmfeature;
     //glrlm feature of tumor save
    float tumorDepth,tumorMeanGray,tumorStdGray,solidity,majorAxisLength,minorAxisLength,tumorArea,tumorPerimeter,bgAvg,tumorNum;
     //肿瘤的所在深度(像素个数为单位长度),平均灰度,灰度标准差,solidity,长轴长度,短轴长度,面积,周长,Otsu算法得到的阈值(前景和背景的阈值),肿瘤的像素个数
    void IntialBaseVar(const contourpoint * plist,const int &plength);
    void getGLRLMFeatuure(const Mat& dst,int gravel,GLRLMValues<float>& glrlmData);
    TextureEValues getGlcmFeature(Mat dst);
public:
    BUSFeature(uchar* data,int width,int height,int format,const contourpoint *plist,const int & length);
    //data 图片内存地址
    //format #define GRAY8 CV_8UC1 8位1通道 #define RGB888 CV_8UC3 8位3通道
    // plist轮廓点数组： struct contourpoint{int x;int y;};x为矩阵的行标 y是矩阵的列标
    //length 轮廓点个数
    BUSFeature(const char *imgpath,const contourpoint *plist,const int & length);
    // 根据原图和红线标注图的构造函数
    // 用来测试，提取的特征对不对。部署时注释掉
    BUSFeature(const char *imgpath,const char *bgpath);
    ~BUSFeature();
    float getDepth();
    //tumor 所在深度
    float getMajorAxisLength();
    //长径
    float getMinorAxisLength();
     //短径
    float getSolidity();
    //solidity
    float getArea();
    //面积
    float getAvgGrayscale();
     //平均灰度
    float getStdGrayscale();
     //灰度标准差
    float getPerimeter();
   //周长
    vector<float>  getEdgeFeature();
   //获得边缘特征
    vector<float>  getInnerEchoFeature();
    //获得内部回声特征
    vector<float>  getNangShiFeature(float nungthresh=0.4,float shithresh=0.9);
       //获得囊实特征
    vector<float>  getPosterEchoFeature();
        //获得后方回声特征
    vector<float>  getShapeFeature();
    //获得型形状特征
    void printAllFeature(string printpath);
    //输入所有的特征值 ,输出的文件路径为 printpath+"-t.txt"
};
#endif
