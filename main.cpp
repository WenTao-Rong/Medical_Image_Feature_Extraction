#include "busfeature.h"
#include "svmClassifier.h"
using namespace cv;
int main()
{
    BUSFeature b=BUSFeature("./data/2.jpg","./data/2-d.jpg");
    cout<<"------------- 2.jpg Feature test----------------------"<<endl;
    cout<<"The result has saved as text file in the source image path"<<endl;
    b.printAllFeature("./data/2.jpg");
    cout<<"depth:"<<b.getDepth()<<endl;
    cout<<"area:"<<b.getArea()<<endl;
    cout<<"tumormeangray:"<<b.getAvgGrayscale()<<endl;
    cout<<"tumorstdgray:"<<b.getStdGrayscale()<<endl;
    cout<<"solidity:"<<b.getSolidity()<<endl;
    cout<<"Perimeter:"<<b.getPerimeter()<<endl;

    vector<float> EdgeFeature,InnerEchoFeature,PosterEchoFeature,ShapeFeature,NangShiFeature;
    EdgeFeature=b.getEdgeFeature();
    InnerEchoFeature=b.getInnerEchoFeature();
    PosterEchoFeature=b.getPosterEchoFeature();
    ShapeFeature=b.getShapeFeature();
    NangShiFeature=b.getNangShiFeature(0.4,0.9);

    cout<<"------------- 2.jpg classification test---------------"<<endl;
    cout<<"-------------Load the model---------------"<<endl;
    struct modelpath mp= {"./model/edge.model","./model/innerecho.model","./model/posterecho.model","./model/nangshi.model","./model/shape.model"};
    struct profilepath pp= {"./model/edgerange","./model/innerechorange","./model/posterechorange","./model/nangshirange","./model/shaperange"};
    svmClassifier model=svmClassifier::getModel(mp,pp);
    cout<<"--------Classification result-------------"<<endl;
    int EdgeLabel= model.getEdgeLabel(EdgeFeature);
    cout<<"edge label: "<<EdgeLabel<<endl;
    int InnerEchoLabel=  model.getInnerEchoLabel(InnerEchoFeature);
    cout<<"innerecho label: "<<InnerEchoLabel<<endl;
    int PosterEchoLabel= model.getPosterEchoLabel(PosterEchoFeature);
    cout<<"posterlabel label: "<<PosterEchoLabel<<endl;
    int ShapeLabel= model.getShapeLabel(ShapeFeature);
    cout<<"shape label: "<<ShapeLabel<<endl;
    int NangShiLabel=model.getNangShiLabel(NangShiFeature);
    cout<<"nangshi label: "<<NangShiLabel<<endl;
    cout<<"-------------Free the model---------------"<<endl;
    model.freeModel();

    return 0;
}
