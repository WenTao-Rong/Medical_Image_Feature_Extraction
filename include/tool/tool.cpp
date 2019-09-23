#include "tool.h"
using namespace cv;
using namespace std;
const int NUMCORES = 1;
float graysdiff(const Mat &labelgray,const  Mat & img_region,int t_len,bool entiretumor)
{
    Mat  innerlabel,outerlabel;
    Mat element = getStructuringElement(MORPH_RECT, Size(t_len, t_len));
    dilate(labelgray,outerlabel,element);
    Mat out=outerlabel-labelgray;
    float avgdiff;
    if(entiretumor)
    {
        avgdiff=abs(mean(img_region,out)[0]-mean(img_region,labelgray)[0]);
    }else{
        erode(labelgray,innerlabel,element);
        avgdiff=abs(mean(img_region,out)[0]-mean(img_region,labelgray-innerlabel)[0]);
    }
    return avgdiff;
};
float grayavgsep(const Mat &labelgray,const Mat & img_region,int t_len)
{
    Mat  innerlabel,outerlabel;
    Mat sh(labelgray.size(),CV_8UC3,Scalar(0,0,0));
    Mat element = getStructuringElement(MORPH_RECT, Size(t_len,t_len));
    erode(labelgray,innerlabel,element);
    Mat in=labelgray-innerlabel;
    dilate(labelgray,outerlabel,element);
    Mat out=outerlabel-labelgray;
    float innernum,outernum,inneravg,outeravg,u,totalvar,intervar,avgsep;
    innernum=sum(in)[0];
    outernum=sum(out)[0];
    outeravg=mean(img_region,out)[0];
    inneravg=mean(img_region,in)[0];
    u=(innernum*inneravg+outernum*outeravg)/(outernum+innernum);
    intervar=(innernum*pow(inneravg-u,2)+outernum*pow(outeravg-u,2))/(innernum+outernum);
    Scalar     mean,stddev;
    meanStdDev ( img_region, mean, stddev,in+out);
    totalvar=pow(stddev.val[0],2);
    avgsep=intervar/totalvar;
    return avgsep;
};
Rect getLargestComponentByBinimage( Mat binary)
{
	CBlobResult res(binary,Mat(),NUMCORES);
	int largestBlobId=0;
	float areatemp=0;
	for (int i=0;i<res.GetNumBlobs();i++)
    {
       CBlob t2 = res.GetBlob(i);

        if(t2.Area(PIXELWISE)>areatemp){
           largestBlobId=i;
           areatemp=t2.Area(PIXELWISE);
        }
    }
    CBlob t=res.GetBlob(largestBlobId);
    Rect bbox = t.GetBoundingBox();
    return bbox;
};

Mat getContourByRedline(const Mat& src)
{
        Mat edge(src.rows,src.cols,CV_8U,Scalar(0));
        for(int i = 0; i < src.rows; i++)
            {
          for(int j = 0; j < src.cols; j++)
            {
            if(src.at<Vec3b>(i, j)[0]<=110 &&src.at<Vec3b>(i, j)[1]<=110 && src.at<Vec3b>(i, j)[2]>=120 ){
                  edge.at<uchar>(i,j)=1;
           }

            }
          }

 t_PointList pl=getLargestComponentPoint(edge);

   for (int i=0;i<edge.rows;i++)
    for(int j=0;j<edge.cols;j++)
    {
    if(find(pl.begin(), pl.end(),Point(j,i) ) == pl.end())
        edge.at<uchar>(i,j)=0;
    }

      vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edge,contours,hierarchy,CV_RETR_CCOMP,CHAIN_APPROX_SIMPLE,Point());
        Mat label(src.rows,src.cols,CV_8U,Scalar(0));
        //从i=1开始，不包括外轮廓
        for(int i=0; i<contours.size(); i++)
        {
            drawContours(label,contours,i, Scalar(1),CV_FILLED,1,hierarchy);
        }
        return label;
}
;
t_PointList getLargestComponentPoint(Mat binary)
{
    CBlobResult res(binary,Mat(),NUMCORES);
    int largestBlobId=0;
    float areatemp=0;
    for (int i=0; i<res.GetNumBlobs(); i++)
    {
        CBlob t2 = res.GetBlob(i);

        if(t2.Area(PIXELWISE)>areatemp)
        {
            largestBlobId=i;
            areatemp=t2.Area(PIXELWISE);
        }
    }
    CBlob t=res.GetBlob(largestBlobId);
    CBlobContour con=t.GetExternalContour();
    t_PointList pp=con.GetContourPoints();
    return pp;
};
void gdfffunderlr( Mat &labelgray,Mat &tumorUnder,Mat& tumorUnderLR,Range& r32,int underheight)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(labelgray,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    CBlobResult res(labelgray,Mat(),NUMCORES);
    int largestBlobId=0;
    float areatemp=0;
    for (int i=0; i<res.GetNumBlobs(); i++)
    {
        CBlob t2 = res.GetBlob(i);

        if(t2.Area(PIXELWISE)>areatemp)
        {
            largestBlobId=i;
            areatemp=t2.Area(PIXELWISE);
        }
    }
    CBlob t=res.GetBlob(largestBlobId);
    CBlobContour con=t.GetExternalContour();
    t_PointList pp=con.GetContourPoints();
    Moments M=moments(contours[0],1);
    int cy = int(M.m01/M.m00);
        int tumorUnderMinx=labelgray.cols;
    int tumorUnderMaxx=0;
    int tumorUnderMaxy=t.MaxY();

    if((tumorUnderMaxy+underheight)>= (labelgray.rows-1))
        underheight=labelgray.rows-tumorUnderMaxy-1;
    for(Point it:pp)
    {
        if(it.y>cy)
        {
            if(it.x<tumorUnderMinx)
                tumorUnderMinx=it.x;
            if(it.x>tumorUnderMaxx)
                tumorUnderMaxx=it.x;
            for (int i=1; i<=underheight; i++)
            {

                tumorUnder.at<uchar>(it.y+i,it.x)=1;
            }
        }
    }
   tumorUnder=tumorUnder.mul(1-labelgray);
    for (int i=tumorUnderMaxy+1; i<=tumorUnderMaxy+underheight; i++)
    {
        for (int j=0; j<tumorUnderMinx; j++)
            tumorUnderLR.at<uchar>(i,j)=1;
        for (int j=tumorUnderMaxx; j<labelgray.cols; j++)
            tumorUnderLR.at<uchar>(i,j)=1;
    }
    float xlengthtemp=(float)tumorUnderMaxx-tumorUnderMinx;
    r32=Range(tumorUnderMinx+(xlengthtemp/6),tumorUnderMinx+(xlengthtemp*5/6));
};


