#include "BUSFeature.h"
#include <vector>
using namespace std;
using namespace cv;
void BUSFeature::IntialBaseVar(const contourpoint * plist,const int &plength)
{
    //initialize rgbimg imgcols imgrows label255 grayimg edge allPoint
    imgcols=rgbimg.cols;
    imgrows=rgbimg.rows;
    label255=Mat(imgrows,imgcols,CV_8UC1,Scalar(0));
    if(rgbimg.empty())
        cerr<<"cant open image,maybe image path invalid"<<endl;
    if(rgbimg.channels()>1)
        cvtColor(rgbimg,grayimg,CV_BGR2GRAY,1);
    else
    {
       grayimg=rgbimg.clone(); ;
    }
    edge=Mat(imgrows,imgcols,CV_8UC1,Scalar(0));
    //reduce ram size of vector
   // cout<<plist->size()<<endl;
    allPoint.reserve(plength);

    for(int i=0; i<plength; i++)
    {
        Point p=Point(plist[i].x,plist[i].y);
        allPoint.push_back(p);
    }

    for(int i=0; i<plength; i++)
    {
        line(edge,allPoint[i],allPoint[((i+1)%plength)],Scalar(255),1,8,0);
    }
    vector<vector<Point>> contours;
    contours.push_back(allPoint);
    vector<Vec4i> hierarchy;
    //fill the Area of tumor boundary to get label255,label
    drawContours(label255,contours,0, Scalar(255),CV_FILLED,8,hierarchy);

    label=label255/255;

    //get MajorAxisLength MinorAxisLength depth area tumorMeanGray tumorStdGray
    RotatedRect box = fitEllipse(allPoint);
    majorAxisLength=(box.size.width>box.size.height)?box.size.width:box.size.height;
    minorAxisLength=(box.size.width>box.size.height)?box.size.height:box.size.width;
    tumorDepth=getLargestComponentByBinimage(label).y;


    tumorArea = contourArea(allPoint);
    Scalar tumormean,tumorstd;
    meanStdDev(grayimg,tumormean,tumorstd,label);

    tumorMeanGray=tumormean[0];
    tumorStdGray=tumorstd[0];


//get solidity perimeter bgavg tumornum
  //  vector< vector<Point> > hull(allPoint.size());
    vector<Point>  hull;
    convexHull(Mat(allPoint), hull, false);
    float  hull_area = contourArea(hull);
    solidity = float(tumorArea)/hull_area;
    tumorPerimeter=arcLength(allPoint,true);
    Mat imgtemp;

    bgAvg=threshold(grayimg,imgtemp,0,255,cv::THRESH_OTSU);
    tumorNum=0;
     for( int i = 0; i < imgrows;i++)
    {
        uchar* data=label.ptr<uchar>(i);
        for( int j = 0; j < imgcols; j++)
        {
            if(data[j]==1 ) tumorNum++;
        }
    }

    Mat tumor=label.mul(grayimg);
    Rect tumorbbox=getLargestComponentByBinimage(tumor);
    tumor=tumor(tumorbbox);
    //get glcm and glrlm feature of tumor
    glcmfeature=getGlcmFeature(tumor);
    getGLRLMFeatuure(tumor,16,glrlmfeature);
}
BUSFeature::BUSFeature(const char *imgpath,const contourpoint * plist,const int &plength)
{
   rgbimg=imread(imgpath);
   //如果输入的图片是有显示设备信息的边框，需要通过以下代码去除，如果没有边框就注释掉
    Mat imggraytemp,imt;
    cvtColor(rgbimg,imggraytemp,CV_BGR2GRAY);
    threshold(imggraytemp,imt,10,255, THRESH_BINARY);
    Rect bbox=getLargestComponentByBinimage(imt);
    Mat rgbimg = rgbimg(bbox);

    IntialBaseVar(plist,plength);
}
BUSFeature::BUSFeature(uchar* data,int width,int height,int format,const contourpoint *plist,const int & plength)
{
    rgbimg=getPicbyRAM(data,width,height,format);
    //如果输入的图片是有显示设备信息的边框，需要通过以下代码去除，如果没有边框就注释掉
    Mat imggraytemp,imt;
    cvtColor(rgbimg,imggraytemp,CV_BGR2GRAY);
    threshold(imggraytemp,imt,10,255, THRESH_BINARY);
    Rect bbox=getLargestComponentByBinimage(imt);
    Mat rgbimg = rgbimg(bbox);

    IntialBaseVar(plist,plength);
}
BUSFeature::BUSFeature(const char *imgpath,const char *bgpath)
{
   Mat rgbgroundth = imread(bgpath);
    if(rgbgroundth.empty())
    {
        fprintf(stderr,"can't open  file %s\n",bgpath);
        exit(1);
    }


    Mat groundth,imt;
    cvtColor(rgbgroundth,groundth,CV_BGR2GRAY);
    threshold(groundth,imt,10,255, THRESH_BINARY);
    Rect bbox=getLargestComponentByBinimage(imt);
    Mat label_region = rgbgroundth(bbox);

    Mat groundth2=getContourByRedline(label_region);

    Mat sourceimg=imread(imgpath);

    if(sourceimg.empty())
    {
        fprintf(stderr,"can't open  file %s\n",imgpath);
        exit(1);
    }
    rgbimg = sourceimg(bbox);
    cvtColor(sourceimg,sourceimg,CV_BGR2GRAY);


    vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(groundth2,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE,Point());
    int plength=contours[0].size();
    contourpoint *plist=new contourpoint[plength];
    for(int i=0;i<plength;i++){
        plist[i].x=contours[0][i].x;
        plist[i].y=contours[0][i].y;
    }
    IntialBaseVar(plist,plength);

}
BUSFeature::~BUSFeature()
{

}
TextureEValues BUSFeature::getGlcmFeature(Mat dst)
{
     Mat dstChannel;
     GLCM glcm;
     TextureEValues EValues;
     glcm.getOneChannel(dst, dstChannel, CHANNEL_B);
     glcm.GrayMagnitude(dstChannel, dstChannel, GRAY_8);
     glcm.CalcuTextureEValue(dstChannel, EValues, 5, GRAY_8);
    return EValues;
};
void BUSFeature::getGLRLMFeatuure(const Mat& dst,int gravel,GLRLMValues<float>& glrlmData)
{
  Mat dst1;
  dst1=dst.clone();
  dst1=dst1/gravel;
  Mat i2(dst1.size(),CV_32FC1);
  dst1.convertTo(i2,CV_32FC1);
  GLRLMFEATURES2D<float> t;
  if(gravel==8)
  {
    vector<float> diffGrey={1,2,3,4,5,6,7,8};
    t.calculateAllGLRLMFEATURES2D(t,i2,diffGrey);
    t.extractGLRLMData2D(glrlmData,t);
  }
    else if(gravel==16)
    {
        vector<float> diffGrey={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        t.calculateAllGLRLMFEATURES2D(t,i2,diffGrey);
        t.extractGLRLMData2D(glrlmData,t);
    }
    else{
        fprintf(stderr,"gravel should be 8 or 16\n");
        exit(1);
    }
};
float   BUSFeature::getDepth()
{
    return tumorDepth;
}

float  BUSFeature::getArea()
{
    return  tumorArea;
}
float  BUSFeature::getAvgGrayscale()
{
    return tumorMeanGray;
}
float  BUSFeature::getPerimeter()
{
    return tumorPerimeter;
}
float BUSFeature::getMajorAxisLength()
{
    return majorAxisLength;
}
float BUSFeature::getMinorAxisLength()
{
    return minorAxisLength;
}
float BUSFeature::getSolidity()
{
    return solidity;
}
float BUSFeature::getStdGrayscale()
{
    return tumorStdGray;
}
vector<float> BUSFeature::getEdgeFeature()
{

    float avgdiff=graysdiff(label,grayimg,15,false);
    float avgsep=grayavgsep(label,grayimg,10);
    vector<float> EdgeFeature={avgdiff,avgsep};
    return EdgeFeature;

}
vector<float>  BUSFeature::getInnerEchoFeature()
{
     float avgdiff=graysdiff(label,grayimg,20,true);
     vector<float> innerEchoFeature={avgdiff,glcmfeature.contrast,glcmfeature.energy,glcmfeature.entropy,glcmfeature.homogenity};
     return innerEchoFeature;
}
vector<float>  BUSFeature::getNangShiFeature(float nungthresh,float shithresh)
{
    float nungnum,shinum;
    float temp=0;
    for( int i = 0; i < imgrows; ++i)
    {
        for( int j = 0; j < imgcols; ++j )
        {
            if(label.at<uchar>(i,j)==1)
            {
                temp=grayimg.at<uchar>(i,j)/bgAvg;
                if(temp<=nungthresh)
                {
                    nungnum=nungnum+1;
                }
                if(temp>=shithresh)
                {
                    shinum=shinum+1;
                }
            }
        }
    }
   vector<float> NangShiFeature={nungnum/tumorNum,shinum/tumorNum,tumorMeanGray/bgAvg,tumorStdGray/bgAvg,
    glcmfeature.contrast,glcmfeature.energy,glcmfeature.entropy,glcmfeature.homogenity,
    glrlmfeature.shortRunEmphasis,glrlmfeature.longRunEmphasis,glrlmfeature.lowGreyEmph,glrlmfeature.highGreyEmph,glrlmfeature.shortRunLow,
    glrlmfeature.shortRunHigh,glrlmfeature.longRunLowEmph,
    glrlmfeature.longRunHighEmph,glrlmfeature.greyNonUniformity,
    glrlmfeature.greyNonUniformityNorm,glrlmfeature.runLengthNonUniformity,glrlmfeature.runLengthNonUniformityNorm
	,glrlmfeature.greyLevelVar,glrlmfeature.runLengthVar,glrlmfeature.runEntropy,glrlmfeature.runPercentage};
    return NangShiFeature;
}
vector<float>  BUSFeature::getPosterEchoFeature()
{

    Mat tumorUnder(label.size(),label.type(),Scalar(0));
    Mat tumorUnderLR(label.size(),label.type(),Scalar(0));
    Range r32;
    gdfffunderlr(label,tumorUnder,tumorUnderLR,r32,100);

    Mat outerlabel;
    Mat element = getStructuringElement(MORPH_RECT,Size(20, 20));
    dilate(label,outerlabel,element);
    Mat out=outerlabel-label;
    float avgtumorsur=mean(grayimg,out)[0];

    float underavg=mean(grayimg,tumorUnder)[0];
   // float ulravg=mean(grayimg,tumorUnderLR)[0];

    float underavg32=mean(grayimg(Range::all(),r32),tumorUnder(Range::all(),r32))[0];
    vector<float> PosterEchoFeature={underavg32-tumorMeanGray,underavg32-avgtumorsur,underavg32-bgAvg,underavg32-underavg};
    return PosterEchoFeature;
}
vector<float>  BUSFeature::getShapeFeature()
{
    vector<Point>  hull;
    vector<int> hullsI;
    vector<Vec4i> defects;
     convexHull( Mat(allPoint), hull, false );
    convexHull(Mat(allPoint), hullsI, false);
    convexityDefects(Mat(allPoint),hullsI, defects);
    float hull_perimeter = arcLength(hull,true);
    vector<Vec4i>::iterator d =defects.begin();
    float concavenum=0;
    float ConcaveAvgDepth=0.0;
    while(d!=defects.end())
    {
        Vec4i& v=(*d);
        float Concavedepth = v[3]/256.0;
        if(Concavedepth > 3.0)
        {
            concavenum++;
            ConcaveAvgDepth=Concavedepth+ConcaveAvgDepth;
        }
        d++;
    }
    if(concavenum>0)
        ConcaveAvgDepth=ConcaveAvgDepth/concavenum;
    float Apect_Radio=4*CV_PI*tumorArea/pow(tumorPerimeter,2);
    float ENC=CV_PI*majorAxisLength*minorAxisLength/(4*tumorPerimeter);
    vector<float> ShapeFeature= {hull_perimeter/tumorPerimeter,solidity,Apect_Radio,ENC,concavenum,ConcaveAvgDepth};
    return ShapeFeature;
}

void BUSFeature::printAllFeature(string printpath)
{

    vector<float>  edgefeature=getEdgeFeature();
    vector<float>  innerEchofeature=getInnerEchoFeature();
    vector<float>  nangShifeature=getNangShiFeature(0.4,0.9);
    vector<float>  posterfeature=getPosterEchoFeature();
    vector<float>  shapefeature=getShapeFeature();
    vector< vector <float>>allfeature={edgefeature,innerEchofeature,posterfeature,nangShifeature,shapefeature};
    ofstream SaveFile(printpath+"-t.txt");
    for(int i=0;i<allfeature.size();i++)
        {
            for(int j=0;j<allfeature[i].size();j++)
            {
                 SaveFile<<" "<<(j+1)<<":"<<allfeature[i][j];
            }
            SaveFile<< "\n";
        }
        SaveFile.close();
    }



