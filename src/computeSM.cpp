//
// Created by lu on 19-4-17.
//

#include "computeSM.h"
#include"triangle.h"
#include"delaunay.h"
#include"Vertex.h"
#include <cmath>
#include <Eigen/Core>
#include<Eigen/Dense>


const double PI = 3.1415926;
const double Pa = 0.5;
const double Ks = PI/2;
const double Ns = 3;

template <class T>
double SimilarityValue(Triangle<float> t1,Triangle<float>t2);


void ComputeSimilarityMatrix(const Delaunay<float> & net1, const Delaunay<float> & net2, Eigen::MatrixXd &similarityMatrix)
{
    std::vector<Triangle<float> > triangleList1(net1.GetTriangles());
    std::vector<Triangle<float> > triangleList2(net2.GetTriangles());


    for (int i = 0; i < triangleList1.size(); i++)
    {

        for (int j = 0; j < triangleList2.size(); j++)
        {


            double a = SimilarityValue(triangleList1[i], triangleList2[j]);
           //std::cout << "a="<< a <<std::endl;
            similarityMatrix(i,j) = (a < 0.7) ? 0 : a; //相似度阈值为0.75
        }
    }
}


double SimilarityValue(Triangle<float> t1,Triangle<float>t2)

{
    double q[3];
    double k[3];
    double dx[3];
    double u[3];
    t1.ComputeAngle();
    t2.ComputeAngle();
    for (int i = 0; i < 3; i++)
    {
//        std::cout << "angle[i] :" <<  t1.angle[i]  << std::endl;
        q[i] = (t1.angle[i] * Pa) / 3;
//        std::cout << "q[i]:" << q[i]  << std::endl;

        k[i] = 1 / (2 * q[i] * q[i]);
//        std::cout << "k[i]:" << k[i]  << std::endl;

        dx[i]= exp(-1 * k[i] * (t2.angle[i] - t1.angle[i]) * (t2.angle[i] - t1.angle[i]));
//        std::cout << "dx[i]:" << dx[i]  << std::endl;

        u[i] = pow(cos(Ks * (1 - dx[i])), Ns);
//        std::cout << "u[i]:" << u[i]  << std::endl;
    }

    return (u[0]+u[1]+u[2]) / 3;
}

double checkconstriant(const Delaunay<float> & net1, const Delaunay<float> & net2,const int t1Index,const int t2Index)
{
    std::vector<Triangle<float> > triangleList1(net1.GetTriangles());
    std::vector<Triangle<float> > triangleList2(net2.GetTriangles());
    cv::Point2f p0,p1,p2,p3,p4,p5;
    p0.x= triangleList1[t1Index].p1.x;
    p0.y= triangleList1[t1Index].p1.y;
    p1.x= triangleList1[t1Index].p2.x;
    p1.y= triangleList1[t1Index].p2.y;
    p2.x= triangleList1[t1Index].p3.x;
    p2.y= triangleList1[t1Index].p3.y;

    p3.x= triangleList2[t2Index].p1.x;
    p3.y= triangleList2[t2Index].p1.y;
    p4.x= triangleList2[t2Index].p2.x;
    p4.y= triangleList2[t2Index].p2.y;
    p5.x= triangleList2[t2Index].p3.x;
    p5.y= triangleList2[t2Index].p3.y;



    std::vector<int>p0EdgeIndex;
    std::vector<int>p1EdgeIndex;
    std::vector<int>p2EdgeIndex;
    std::vector<int>p3EdgeIndex;
    std::vector<int>p4EdgeIndex;
    std::vector<int>p5EdgeIndex;


    std::vector<Edge<float>> edgelist1(net1.GetEdges());
    std::vector<Edge<float>> edgelist2(net2.GetEdges());

    for (int i = 0; i <edgelist1.size() ; ++i)
    {
        if((edgelist1[i].p1.x==p0.x && edgelist1[i].p1.y==p0.y)||(edgelist1[i].p2.x==p0.x  && edgelist1[i].p2.y==p0.y))
        {
            p0EdgeIndex.push_back(i);

        }
        if((edgelist1[i].p1.x==p1.x && edgelist1[i].p1.y==p1.y)||(edgelist1[i].p2.x==p1.x  && edgelist1[i].p2.y==p1.y))
        {
            p1EdgeIndex.push_back(i);

        }
        if((edgelist1[i].p1.x==p2.x && edgelist1[i].p1.y==p2.y)||(edgelist1[i].p2.x==p2.x  && edgelist1[i].p2.y==p2.y))
        {
            p2EdgeIndex.push_back(i);

        }
    }
    cout<<"p0EdgeIndex:"<<p0EdgeIndex.size()<<endl;
    for (int k = 0; k <p0EdgeIndex.size() ; ++k)
    {
        cout<<"  p0:"<<p0EdgeIndex[k]<<endl;

    }
    for (int i = 0; i <edgelist2.size() ; ++i)
    {
        if((edgelist2[i].p1.x==p3.x && edgelist2[i].p1.y==p3.y)||(edgelist2[i].p2.x==p3.x  && edgelist2[i].p2.y==p3.y))
        {
            p3EdgeIndex.push_back(i);

        }
        if((edgelist2[i].p1.x==p4.x && edgelist2[i].p1.y==p4.y)||(edgelist2[i].p2.x==p4.x  && edgelist2[i].p2.y==p4.y))
        {
            p4EdgeIndex.push_back(i);

        }
        if((edgelist2[i].p1.x==p5.x && edgelist2[i].p1.y==p5.y)||(edgelist2[i].p2.x==p5.x  && edgelist2[i].p2.y==p5.y))
        {
            p5EdgeIndex.push_back(i);

        }
    }

    std::vector<cv::Point2f> outP(6);
    for (int i = 0; i <p0EdgeIndex.size() ; ++i)
    {
        for (int j = 0; j < p1EdgeIndex.size(); ++j)
        {
            cv::Vec4f e1,e2;
            e1[0]=edgelist1[i].p1.x;e1[1]=edgelist1[i].p1.y;
            e1[2]=edgelist1[i].p2.x;e1[3]=edgelist1[i].p2.y;

            e2[0]=edgelist1[j].p1.x;e2[1]=edgelist1[j].p1.y;
            e2[2]=edgelist1[j].p2.x;e2[3]=edgelist1[j].p2.y;


            if (e1[0]==e2[0]&&e1[1]==e2[1])
            {
                outP[0].x=e1[0];
                outP[0].y=e1[1];

            }
            if (e1[0]==e2[2]&&e1[1]==e2[3])
            {
                outP[0].x=e1[0];
                outP[0].y=e1[1];

            }
            if (e1[2]==e2[0]&&e1[3]==e2[1])
            {
                outP[0].x=e1[2];
                outP[0].y=e1[3];

            }
            if (e1[2]==e2[2]&&e1[3]==e2[3])
            {
                outP[0].x=e1[2];
                outP[0].y=e1[3];

            }

        }

    }
    for (int i = 0; i <p0EdgeIndex.size() ; ++i)
    {
        for (int j = 0; j < p2EdgeIndex.size(); ++j)
        {
            cv::Vec4f e1,e2;
            e1[0]=edgelist1[i].p1.x;e1[1]=edgelist1[i].p1.y;
            e1[2]=edgelist1[i].p2.x;e1[3]=edgelist1[i].p2.y;

            e2[0]=edgelist1[j].p1.x;e2[1]=edgelist1[j].p1.y;
            e2[2]=edgelist1[j].p2.x;e2[3]=edgelist1[j].p2.y;


            if (e1[0]==e2[0]&&e1[1]==e2[1])
            {
                outP[1].x=e1[0];
                outP[1].y=e1[1];

            }
            if (e1[0]==e2[2]&&e1[1]==e2[3])
            {
                outP[1].x=e1[0];
                outP[1].y=e1[1];

            }
            if (e1[2]==e2[0]&&e1[3]==e2[1])
            {
                outP[1].x=e1[2];
                outP[1].y=e1[3];

            }
            if (e1[2]==e2[2]&&e1[3]==e2[3])
            {
                outP[1].x=e1[2];
                outP[1].y=e1[3];

            }

        }

    }
    for (int i = 0; i <p1EdgeIndex.size() ; ++i)
    {
        for (int j = 0; j < p2EdgeIndex.size(); ++j)
        {
            cv::Vec4f e1,e2;
            e1[0]=edgelist1[i].p1.x;e1[1]=edgelist1[i].p1.y;
            e1[2]=edgelist1[i].p2.x;e1[3]=edgelist1[i].p2.y;

            e2[0]=edgelist1[j].p1.x;e2[1]=edgelist1[j].p1.y;
            e2[2]=edgelist1[j].p2.x;e2[3]=edgelist1[j].p2.y;


            if (e1[0]==e2[0]&&e1[1]==e2[1])
            {
                outP[2].x=e1[0];
                outP[2].y=e1[1];

            }
            if (e1[0]==e2[2]&&e1[1]==e2[3])
            {
                outP[2].x=e1[0];
                outP[2].y=e1[1];

            }
            if (e1[2]==e2[0]&&e1[3]==e2[1])
            {
                outP[2].x=e1[2];
                outP[2].y=e1[3];

            }
            if (e1[2]==e2[2]&&e1[3]==e2[3])
            {
                outP[2].x=e1[2];
                outP[2].y=e1[3];

            }

        }

    }
    for (int i = 0; i <p3EdgeIndex.size() ; ++i)
    {
        for (int j = 0; j < p4EdgeIndex.size(); ++j)
        {
            cv::Vec4f e1,e2;
            e1[0]=edgelist2[i].p1.x;e1[1]=edgelist2[i].p1.y;
            e1[2]=edgelist2[i].p2.x;e1[3]=edgelist2[i].p2.y;

            e2[0]=edgelist2[j].p1.x;e2[1]=edgelist2[j].p1.y;
            e2[2]=edgelist2[j].p2.x;e2[3]=edgelist2[j].p2.y;


            if (e1[0]==e2[0]&&e1[1]==e2[1])
            {
                outP[3].x=e1[0];
                outP[3].y=e1[1];

            }
            if (e1[0]==e2[2]&&e1[1]==e2[3])
            {
                outP[3].x=e1[0];
                outP[3].y=e1[1];

            }
            if (e1[2]==e2[0]&&e1[3]==e2[1])
            {
                outP[3].x=e1[2];
                outP[3].y=e1[3];

            }
            if (e1[2]==e2[2]&&e1[3]==e2[3])
            {
                outP[3].x=e1[2];
                outP[3].y=e1[3];

            }

        }

    }
    for (int i = 0; i <p3EdgeIndex.size() ; ++i)
    {
        for (int j = 0; j < p5EdgeIndex.size(); ++j)
        {
            cv::Vec4f e1,e2;
            e1[0]=edgelist2[i].p1.x;e1[1]=edgelist2[i].p1.y;
            e1[2]=edgelist2[i].p2.x;e1[3]=edgelist2[i].p2.y;

            e2[0]=edgelist2[j].p1.x;e2[1]=edgelist2[j].p1.y;
            e2[2]=edgelist2[j].p2.x;e2[3]=edgelist2[j].p2.y;


            if (e1[0]==e2[0]&&e1[1]==e2[1])
            {
                outP[4].x=e1[0];
                outP[4].y=e1[1];

            }
            if (e1[0]==e2[2]&&e1[1]==e2[3])
            {
                outP[4].x=e1[0];
                outP[4].y=e1[1];

            }
            if (e1[2]==e2[0]&&e1[3]==e2[1])
            {
                outP[4].x=e1[2];
                outP[4].y=e1[3];

            }
            if (e1[2]==e2[2]&&e1[3]==e2[3])
            {
                outP[4].x=e1[2];
                outP[4].y=e1[3];

            }

        }

    }
    for (int i = 0; i <p4EdgeIndex.size() ; ++i)
    {
        for (int j = 0; j < p5EdgeIndex.size(); ++j)
        {
            cv::Vec4f e1,e2;
            e1[0]=edgelist2[i].p1.x;e1[1]=edgelist2[i].p1.y;
            e1[2]=edgelist2[i].p2.x;e1[3]=edgelist2[i].p2.y;

            e2[0]=edgelist2[j].p1.x;e2[1]=edgelist2[j].p1.y;
            e2[2]=edgelist2[j].p2.x;e2[3]=edgelist2[j].p2.y;


            if (e1[0]==e2[0]&&e1[1]==e2[1])
            {
                outP[5].x=e1[0];
                outP[5].y=e1[1];

            }
            if (e1[0]==e2[2]&&e1[1]==e2[3])
            {
                outP[5].x=e1[0];
                outP[5].y=e1[1];

            }
            if (e1[2]==e2[0]&&e1[3]==e2[1])
            {
                outP[5].x=e1[2];
                outP[5].y=e1[3];

            }
            if (e1[2]==e2[2]&&e1[3]==e2[3])
            {
                outP[5].x=e1[2];
                outP[5].y=e1[3];

            }

        }

    }

   //这里得到三个Point2f类型的点
    Vertex<float> _p0(p0.x,p0.y);
    Vertex<float> _p1(p1.x,p1.y);
    Vertex<float> _p2(outP[0].x,outP[0].y);
    Triangle<float> outT1(_p0,_p1,_p2);
    Vertex<float> _p3(p3.x,p3.y);
    Vertex<float> _p4(p4.x,p4.y);
    Vertex<float> _p5(outP[3].x,outP[3].y);
    Triangle<float> _outT1(_p3,_p4,_p5);

    Vertex<float> p_0(p0.x,p0.y);
    Vertex<float> p_1(p2.x,p2.y);
    Vertex<float> p_2(outP[1].x,outP[1].y);
    Triangle<float> outT_1(p_0,p_1,p_2);
    Vertex<float> _p_0(p3.x,p3.y);
    Vertex<float> _p_1(p5.x,p5.y);
    Vertex<float> _p_2(outP[4].x,outP[4].y);
    Triangle<float> outT_2(_p_0,_p_1,_p_2);

    Vertex<float> p0_(p1.x,p1.y);
    Vertex<float> p1_(p2.x,p2.y);
    Vertex<float> p2_(outP[2].x,outP[2].y);
    Triangle<float> outT_3(p0_,p1_,p2_);
    Vertex<float> p2_1(p4.x,p4.y);
    Vertex<float> p2_2(p5.x,p5.y);
    Vertex<float> p2_3(outP[5].x,outP[5].y);
    Triangle<float> outT_4(p2_1,p2_2,p2_3);

    double similrity1=SimilarityValue(outT1,_outT1);
    double similrity2=SimilarityValue(outT_1,outT_2);
    double similrity3=SimilarityValue(outT_3,outT_4);
    if(similrity1<0.75||similrity2<0.75||similrity3<0.75)
    {
        return 0;
    }

    return 1;

}
