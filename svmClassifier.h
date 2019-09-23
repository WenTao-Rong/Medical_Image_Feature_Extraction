#ifndef _SVMCLASSIFIER_
#define _SVMCLASSIFIER_
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <vector>
#include <fstream>
#include "./include/svm/svm.h"

//定义特征的长度
#define EdgeFeaturelen 2
#define InnerEchoFeaturelen 5
#define PosterEchoFeaturelen 4
#define ShapeFeaturelen 6
#define NangshiFeaturelen 24
using namespace std;
static int (*info)(const char *fmt,...) = &printf;

//模型文件路径
typedef struct modelpath
{
    char * edge;
    char * innerecho;
    char * posterecho;
    char * nangshi;
    char * shape;
} modelpath;
//配置文件路径
struct profilepath
{
    char * edge;
    char * innerecho;
    char * posterecho;
    char * nangshi;
    char * shape;
};
struct featurerange
{
    float minf;
    float maxf;
};

class svmClassifier
{
    private:
     struct svm_model*  EdgeModel;
     struct svm_model*  InnerEchoModel;
     struct svm_model*  NangShiModel;
     struct svm_model*  PosterEchoModel;
     struct svm_model*  ShapeModel;
     struct featurerange EdgeRange[EdgeFeaturelen],NangShiRange[NangshiFeaturelen],PosterEchoRange[PosterEchoFeaturelen], ShapeRange[ShapeFeaturelen],InnerEchoRange[InnerEchoFeaturelen];
     svmClassifier(struct modelpath & model,struct profilepath & path);
public:
//加载模型后加载配置文件
    static svmClassifier & getModel(struct modelpath & model,struct profilepath & path)
    {
        static svmClassifier instance=svmClassifier(model,path);  //局部静态变量
        return instance;
    }
  //  ~svmClassifier();
    void loadModel(struct modelpath & model);
    void loadProfile(struct profilepath & path);
//系统关闭时释放内存
     void freeModel();

    void scaleFeature(const vector<float> & feature,vector<float> & scaledfeature,struct featurerange * range);
//预测函数
    float predictLabel(const vector<float> & attrlist,struct svm_model *model);
    int   getEdgeLabel(const vector<float> &feature);
    // 边界清晰度label 1 为清晰 -1为模糊
    int   getInnerEchoLabel(const vector<float> & feature);
    //内部回声label 1 低回声 2无回声 3 混合回声
    int   getNangShiLabel(const vector<float> &feature);
    //囊实状态 1 为囊性 2 为实性 3 混合
    int   getPosterEchoLabel(const vector<float> &feature);
    //后方回声label 1 为衰减 2不变 3 增强
    int   getShapeLabel(const vector<float> &feature);

};


#endif

