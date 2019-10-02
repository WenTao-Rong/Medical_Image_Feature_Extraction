# 超声图像分析系统（分割，特征提取，分类）

 使用的框架和开发语言是 opencv 3.4.5 和c++
 * 使用GVF snake模型或深度学习模型分割
 * 提取超声图像特征，包括形状，边缘，囊实，内部回声，后方回声特征
 * 使用分类器（SVM 或逻辑回归）得到医生注重的肿瘤特征描述
 
 
 # 快速开始
 
1.运行main.cpp

2.比对结果,我的结果在./data/2.jpg提取的特征.txt 和./data/2.jpg运行结果.jpg

3.如果结果一致,根据情况注释以下代码
* 将busfeature.h busfeature.cpp里的void printAllFeature(string printpath)
* busfeature.h 两个构造函数BUSFeature::BUSFeature(const char *imgpath,const contourpoint * plist,const int &plength)和 BUSFeature::BUSFeature(uchar* data,int width,int height,int format,const contourpoint *plist,const int & plength)
中的以下代码：

    //如果输入的图片是没有设备信息的边框，就注释以下代码，反之保留
    
	
    Mat imggraytemp,imt;
    cvtColor(rgbimg,imggraytemp,CV_BGR2GRAY);
    threshold(imggraytemp,imt,10,255, THRESH_BINARY);
    Rect bbox=getLargestComponentByBinimage(imt);
    Mat rgbimg = rgbimg(bbox);
	
4.如果在window环境运行，需要配置pthread，如果在linux环境下，ComponentLabeling.h 不用 #include "../pthread/pthread.h"，

 # 特征提取
 
| 特征类别 | 具体特征 | 特征描述 |
| :--------------- |:---------------:| ---------------:|
| 形状  |凸度, 坚固度, 形状紧凑度, 椭圆归一化圆周, <br> 凸包缺陷检测的最远点的个数和平均长度|圆, 椭圆, 不规则|
|内部回声|肿瘤内部与外部带状区域的灰度差, GLCM特征 | 衰减, 无改变, 增强 |
|后方回声 |肿瘤下方区域与肿瘤内部的灰度差, 肿瘤下方区域与肿瘤周围的灰度差,<br> 肿瘤下方区域与OSTU阈值的灰度差| 衰减, 无改变, 增强|
|囊实  | 囊性比例, 实性比例, 肿瘤内部平均灰度/OSTU阈值, <br>肿瘤内部灰度标准差/OSTU阈值, GLCM特征,GLRLM特征 |囊, 实, 混合|
|边缘  | 肿瘤内外带状区域的灰度差, 肿瘤内外带状区域的类间方差  |清晰, 模糊|

## 形状特征

* **凸度** $$=\frac{ConvexPerimeter}{Perimeter} \quad$$  其中, ConvexPerimeter是肿瘤凸包的周长, Perimeter是肿瘤的周长

* **坚固度（solidity）** $$=\frac{TumorArea}{ConvexArea} \quad$$ 其中, TumorArea是肿瘤的面积, ConvexArea是肿瘤凸包的面积

* **椭圆归一化圆周(ENC:elliptic-normalized circumference)**$$=\frac{EquivalentEllipsePerimeter}{Perimeter} \quad$$ 其中, EquivalentEllipsePerimeter 是肿瘤的等效椭圆周长

* **凸包缺陷检测的最远点的个数和平均长度**

![](https://github.com/WenTao-Rong/Medical_Image_Feature_Extraction/raw/master/doc/Convex_Test.jpg)

图中红色轮廓是肿瘤的真实轮廓, 蓝点表示凸包检测(使用opencv的convexityDefects 函数)每一个凸包缺陷区域的开始点或结束点，将这些点连起来就会得到一个凸包(绿色轮廓),红点是凸包检测返回的每个凸包缺陷区域中距离凸包最远的点（展示和实际使用的最远点的距离都大于3个像素点） 

## 内部回声特征

* **肿瘤内部与外部带状区域的灰度差**

![](https://github.com/WenTao-Rong/Medical_Image_Feature_Extraction/raw/master/doc/TumorInnerOutter1.jpg)

红色区域表示肿瘤内部,绿色区域表示肿瘤外部20个像素长的带状区域, 灰度差等于肿瘤外部带状区域的平均灰度-肿瘤内部的平均灰度

* **GLCM特征**：能量, 对比度, 相关度, 熵 

## 边缘特征

* **肿瘤内外带状区域的灰度差**

![](https://github.com/WenTao-Rong/Medical_Image_Feature_Extraction/raw/master/doc/TumorInnerOutter2.jpg)

红色区域和绿色区域表示肿瘤内外15个像素长的带状区域, 灰度差等于肿瘤外部带状区域的平均灰度-肿瘤内带状区域的平均灰度

## 后方回声特征

## 囊实特征


