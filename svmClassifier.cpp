#include "svmClassifier.h"
int max_nr_attr = 64;

void svmClassifier::loadModel(struct modelpath & model)
{
    if((EdgeModel=svm_load_model(model.edge))==0)
    {
        fprintf(stderr,"can't open model file %s\n",model.edge);
        exit(1);
    }
    if((InnerEchoModel=svm_load_model(model.innerecho))==0)
    {
        fprintf(stderr,"can't open model file %s\n",model.innerecho);
        exit(1);
    }
    if((PosterEchoModel=svm_load_model(model.posterecho))==0)
    {
        fprintf(stderr,"can't open model file %s\n",model.posterecho);
        exit(1);
    }
    if((NangShiModel=svm_load_model(model.nangshi))==0)
    {
        fprintf(stderr,"can't open model file %s\n",model.nangshi);
        exit(1);
    }
    if((ShapeModel=svm_load_model(model.shape))==0)
    {
        fprintf(stderr,"can't open model file %s\n",model.shape);
        exit(1);
    }
};
void svmClassifier::freeModel()
{
    svm_free_and_destroy_model(&EdgeModel);
    svm_free_and_destroy_model(&InnerEchoModel);
    svm_free_and_destroy_model(&PosterEchoModel);
    svm_free_and_destroy_model(&NangShiModel);
    svm_free_and_destroy_model(&ShapeModel);
};
void svmClassifier::loadProfile(struct profilepath & profile)
{
    ifstream range;
    range.open(profile.nangshi);
    string p;
    int index;
    float minvalue,maxvalue;
    while(getline(range,p))
    {
        const char *cstr = p.c_str();
        sscanf(cstr,"%d:%f:%f\n",&index,&minvalue,&maxvalue);
        NangShiRange[(index-1)].minf=minvalue;
        NangShiRange[(index-1)].maxf=maxvalue;
    }
    range.close();

    ifstream range1;
    range1.open(profile.edge);
    while(getline(range1,p))
    {
        const char *cstr = p.c_str();
        sscanf(cstr,"%d:%f:%f\n",&index,&minvalue,&maxvalue);
        EdgeRange[(index-1)].minf=minvalue;
        EdgeRange[(index-1)].maxf=maxvalue;
    }
    range1.close();

    ifstream range2;
    range2.open(profile.innerecho);
    while(getline(range2,p))
    {
        const char *cstr = p.c_str();
        sscanf(cstr,"%d:%f:%f\n",&index,&minvalue,&maxvalue);
        InnerEchoRange[(index-1)].minf=minvalue;
       InnerEchoRange[(index-1)].maxf=maxvalue;
    }
    range2.close();

    ifstream range3;
    range3.open(profile.posterecho);
    while(getline(range3,p))
    {
        const char *cstr = p.c_str();
        sscanf(cstr,"%d:%f:%f\n",&index,&minvalue,&maxvalue);
        PosterEchoRange[(index-1)].minf=minvalue;
       PosterEchoRange[(index-1)].maxf=maxvalue;
    }
    range3.close();


    ifstream range4;
    range4.open(profile.shape);

    while(getline(range4,p))
    {
        const char *cstr = p.c_str();
        sscanf(cstr,"%d:%f:%f\n",&index,&minvalue,&maxvalue);
        ShapeRange[(index-1)].minf=minvalue;
       ShapeRange[(index-1)].maxf=maxvalue;
    }
    range4.close();


};
svmClassifier::svmClassifier(struct modelpath & model,struct profilepath & path)
{
   loadModel(model);
   loadProfile(path);
};
void  svmClassifier::scaleFeature(const vector<float> & feature,vector<float> & scaledfeature,struct featurerange * range)
{
    for (int i=0; i<feature.size(); i++)
    {
        scaledfeature.push_back((feature[i]-range[i].minf)/(range[i].maxf-range[i].minf));
    }
}
;
float svmClassifier::predictLabel(const vector<float> & attrlist,struct svm_model* model)
{
	struct svm_node *x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));

    unsigned i=0;
    while(i<attrlist.size()){
             x[i].index = i+1;
			x[i].value = attrlist[i];
        i++;
	}
    x[i].index = -1;
    if(model == NULL)
    {
        fprintf(stderr,"%s","model null");
    }
    float predict_label = svm_predict(model,x);
	free(x);
	return predict_label;

};
int  svmClassifier::getEdgeLabel(const vector<float> & feature )
{
     vector<float> scaledfeature;
    scaledfeature.reserve(feature.size());
    scaleFeature(feature,scaledfeature,EdgeRange);
    int edgelabel=predictLabel(feature,EdgeModel);
    return edgelabel;
}
int   svmClassifier::getInnerEchoLabel(const vector<float> & feature)
{
    vector<float> scaledfeature;
    scaledfeature.reserve(feature.size());
    scaleFeature(feature,scaledfeature,InnerEchoRange);
    int innerlabel=predictLabel(scaledfeature,InnerEchoModel);
    return innerlabel;
}
int svmClassifier::getNangShiLabel(const vector<float> & feature)
{
   // vector<float> feature=getNangShiFeature(0.4,0.9);
    vector<float> scaledfeature;
    scaledfeature.reserve(feature.size());
    scaleFeature(feature,scaledfeature,NangShiRange);
    int nangshilabel=predictLabel(scaledfeature,NangShiModel);
    return nangshilabel;
}


int  svmClassifier::getPosterEchoLabel(const vector<float> & feature)
{
    vector<float> scaledfeature;
    scaledfeature.reserve(feature.size());
    scaleFeature(feature,scaledfeature,PosterEchoRange);
    int posterecholabel=predictLabel(scaledfeature,PosterEchoModel);
    return posterecholabel;
}

int  svmClassifier::getShapeLabel(const vector<float> & feature)
{
    vector<float> scaledfeature;
    scaledfeature.reserve(feature.size());
    scaleFeature(feature,scaledfeature,ShapeRange);
    int shapelabel=predictLabel(scaledfeature,ShapeModel);
//    if(shapelabel==1){
//        if((majorAxisLength/minorAxisLength)<1.2)
//            shapelabel=2;
//    }
    return shapelabel;
}

