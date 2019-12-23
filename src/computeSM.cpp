//
// Created by lu on 19-4-17.
//

//import numpy as np
//import matplotlib.pyplot as plt

#include "computeSM.h"
#include"triangle.h"
#include"delaunay.h"
#include"Vertex.h"
#include <cmath>
#include <Eigen/Core>
#include<Eigen/Dense>
#include<cmath>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include<iostream>

using namespace cv;

const double PI = 4.0*atan(1.0f);
const double Pa = 0.5;
const double Ks = PI/2;
const double Ns = 3;

template <class T>
double SimilarityValue(Triangle<float> t1,Triangle<float>t2);
double ComEntropy(Mat img1, Mat img2, double img1_entropy, double img2_entropy);




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
           // similarityMatrix(i,j) = (a < 0.97) ? 0 : a; //相似度阈值为0.75
            if (a >= 0.75)
            {
                Vertex<float> A(triangleList1[i].mainpoint.x - triangleList1[i].circum.x, triangleList1[i].mainpoint.y - triangleList1[i].circum.y);
                Vertex<float> B(triangleList2[i].mainpoint.x - triangleList2[i].circum.x, triangleList2[i].mainpoint.y - triangleList2[i].circum.y);

                if (A.x != 0 && A.y != 0 && B.x != 0 && B.y != 0) {
                    double cos_value = (A.x*B.x+A.y*B.y)/(sqrt(A.x*A.x+A.y*A.y)*sqrt(B.x*B.x+B.y*B.y));
                    double angle = acos(cos_value);

//                    std::cout << "A(" << A.x << " , " << A.y << ")  B(" << B.x << " , " << B.y << ")" << std::endl;
//                    std::cout << "cos: " << cos_value << " ,Angle: " << angle << std::endl;
                    if (angle < 1.9)    //
                        similarityMatrix(i,j) = a;
                    else
                        similarityMatrix(i,j) = 0;

                }
            }

        }
    }


    fstream outputFile1;
    outputFile1.open("a.txt",std::ios::out);
    int r=similarityMatrix.rows();
    int c=similarityMatrix.cols();

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            outputFile1 << similarityMatrix(i, j)<<" ";
        }
        outputFile1<<endl;
        outputFile1<<endl;
    }
    outputFile1.close();



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
//       std::cout << "u[i]:" << u[i]  << std::endl;
    }

    return (u[0]+u[1]+u[2]) / 3;
}






///*
//double checkconstriant(const Delaunay<float> & net1, const Delaunay<float> & net2,const int t1Index,const int t2Index)
//{
//    std::vector<Triangle<float> > triangleList1(net1.GetTriangles());
//    std::vector<Triangle<float> > triangleList2(net2.GetTriangles());
//    Vertex<float> p0,p1,p2,p3,p4,p5;
//    p0.x= triangleList1[t1Index].p1.x;
//    p0.y= triangleList1[t1Index].p1.y;
//    p0.index=triangleList1[t1Index].p1.index;
//    p1.x= triangleList1[t1Index].p2.x;
//    p1.y= triangleList1[t1Index].p2.y;
//    p1.index=triangleList1[t1Index].p2.index;
//    p2.x= triangleList1[t1Index].p3.x;
//    p2.y= triangleList1[t1Index].p3.y;
//    p2.index=triangleList1[t1Index].p3.index;
//
//    p3.x= triangleList2[t2Index].p1.x;
//    p3.y= triangleList2[t2Index].p1.y;
//    p3.index=triangleList1[t2Index].p1.index;
//    p4.x= triangleList2[t2Index].p2.x;
//    p4.y= triangleList2[t2Index].p2.y;
//    p4.index=triangleList1[t2Index].p2.index;
//    p5.x= triangleList2[t2Index].p3.x;
//    p5.y= triangleList2[t2Index].p3.y;
//    p5.index=triangleList1[t2Index].p3.index;
//
//
//
//    std::vector<int>p0EdgeIndex;
//    std::vector<int>p1EdgeIndex;
//    std::vector<int>p2EdgeIndex;
//    std::vector<int>p3EdgeIndex;
//    std::vector<int>p4EdgeIndex;
//    std::vector<int>p5EdgeIndex;
//
//
//    std::vector<Edge<float>> edgelist1(net1.GetEdges());
//    std::vector<Edge<float>> edgelist2(net2.GetEdges());
//
//    for (int i = 0; i <edgelist1.size() ; i++)
//    {
//        if((edgelist1[i].p1.x==p0.x && edgelist1[i].p1.y==p0.y)||(edgelist1[i].p2.x==p0.x  && edgelist1[i].p2.y==p0.y))
//        {
//            p0EdgeIndex.push_back(i);
//
//        }
//        if((edgelist1[i].p1.x==p1.x && edgelist1[i].p1.y==p1.y)||(edgelist1[i].p2.x==p1.x  && edgelist1[i].p2.y==p1.y))
//        {
//            p1EdgeIndex.push_back(i);
//
//        }
//        if((edgelist1[i].p1.x==p2.x && edgelist1[i].p1.y==p2.y)||(edgelist1[i].p2.x==p2.x  && edgelist1[i].p2.y==p2.y))
//        {
//            p2EdgeIndex.push_back(i);
//
//        }
//    }
//    //cout<<"p0EdgeIndex:"<<p0EdgeIndex.size()<<endl<<"p1EdgeIndex:"<<p1EdgeIndex.size()<<endl<<"p2EdgeIndex:"<<p2EdgeIndex.size()<<endl;
//
//
//
//    for (int i = 0; i <edgelist2.size() ; i++)
//    {
//        if((edgelist2[i].p1.x==p3.x && edgelist2[i].p1.y==p3.y)||(edgelist2[i].p2.x==p3.x  && edgelist2[i].p2.y==p3.y))
//        {
//            p3EdgeIndex.push_back(i);
//
//        }
//        if((edgelist2[i].p1.x==p4.x && edgelist2[i].p1.y==p4.y)||(edgelist2[i].p2.x==p4.x  && edgelist2[i].p2.y==p4.y))
//        {
//            p4EdgeIndex.push_back(i);
//
//        }
//        if((edgelist2[i].p1.x==p5.x && edgelist2[i].p1.y==p5.y)||(edgelist2[i].p2.x==p5.x  && edgelist2[i].p2.y==p5.y))
//        {
//            p5EdgeIndex.push_back(i);
//
//        }
//    }
//
//    std::vector<cv::Point2f> outP(6);
//    for (int i = 0; i <p0EdgeIndex.size() ; ++i)
//    {
//        for (int j = 0; j < p1EdgeIndex.size(); ++j)
//        {
//            cv::Vec4f e1,e2;
//            e1[0]=edgelist1[i].p1.x;e1[1]=edgelist1[i].p1.y;
//            e1[2]=edgelist1[i].p2.x;e1[3]=edgelist1[i].p2.y;
//
//            e2[0]=edgelist1[j].p1.x;e2[1]=edgelist1[j].p1.y;
//            e2[2]=edgelist1[j].p2.x;e2[3]=edgelist1[j].p2.y;
//
//
//            if (e1[0]==e2[0]&&e1[1]==e2[1])
//            {
//                outP[0].x=e1[0];
//                outP[0].y=e1[1];
//
//            }
//            if (e1[0]==e2[2]&&e1[1]==e2[3])
//            {
//                outP[0].x=e1[0];
//                outP[0].y=e1[1];
//
//            }
//            if (e1[2]==e2[0]&&e1[3]==e2[1])
//            {
//                outP[0].x=e1[2];
//                outP[0].y=e1[3];
//
//            }
//            if (e1[2]==e2[2]&&e1[3]==e2[3])
//            {
//                outP[0].x=e1[2];
//                outP[0].y=e1[3];
//
//            }
//
//        }
//
//    }
//
//
//    for (int i = 0; i <p0EdgeIndex.size() ; ++i)
//    {
//        for (int j = 0; j < p2EdgeIndex.size(); ++j)
//        {
//            cv::Vec4f e1,e2;
//            e1[0]=edgelist1[i].p1.x;e1[1]=edgelist1[i].p1.y;
//            e1[2]=edgelist1[i].p2.x;e1[3]=edgelist1[i].p2.y;
//
//            e2[0]=edgelist1[j].p1.x;e2[1]=edgelist1[j].p1.y;
//            e2[2]=edgelist1[j].p2.x;e2[3]=edgelist1[j].p2.y;
//
//
//            if (e1[0]==e2[0]&&e1[1]==e2[1])
//            {
//                outP[1].x=e1[0];
//                outP[1].y=e1[1];
//
//            }
//            if (e1[0]==e2[2]&&e1[1]==e2[3])
//            {
//                outP[1].x=e1[0];
//                outP[1].y=e1[1];
//
//            }
//            if (e1[2]==e2[0]&&e1[3]==e2[1])
//            {
//                outP[1].x=e1[2];
//                outP[1].y=e1[3];
//
//            }
//            if (e1[2]==e2[2]&&e1[3]==e2[3])
//            {
//                outP[1].x=e1[2];
//                outP[1].y=e1[3];
//
//            }
//
//        }
//
//    }
//    for (int i = 0; i <p1EdgeIndex.size() ; ++i)
//    {
//        for (int j = 0; j < p2EdgeIndex.size(); ++j)
//        {
//            cv::Vec4f e1,e2;
//            e1[0]=edgelist1[i].p1.x;e1[1]=edgelist1[i].p1.y;
//            e1[2]=edgelist1[i].p2.x;e1[3]=edgelist1[i].p2.y;
//
//            e2[0]=edgelist1[j].p1.x;e2[1]=edgelist1[j].p1.y;
//            e2[2]=edgelist1[j].p2.x;e2[3]=edgelist1[j].p2.y;
//
//
//            if (e1[0]==e2[0]&&e1[1]==e2[1])
//            {
//                outP[2].x=e1[0];
//                outP[2].y=e1[1];
//
//            }
//            if (e1[0]==e2[2]&&e1[1]==e2[3])
//            {
//                outP[2].x=e1[0];
//                outP[2].y=e1[1];
//
//            }
//            if (e1[2]==e2[0]&&e1[3]==e2[1])
//            {
//                outP[2].x=e1[2];
//                outP[2].y=e1[3];
//
//            }
//            if (e1[2]==e2[2]&&e1[3]==e2[3])
//            {
//                outP[2].x=e1[2];
//                outP[2].y=e1[3];
//
//            }
//
//        }
//
//    }
//    for (int i = 0; i <p3EdgeIndex.size() ; ++i)
//    {
//        for (int j = 0; j < p4EdgeIndex.size(); ++j)
//        {
//            cv::Vec4f e1,e2;
//            e1[0]=edgelist2[i].p1.x;e1[1]=edgelist2[i].p1.y;
//            e1[2]=edgelist2[i].p2.x;e1[3]=edgelist2[i].p2.y;
//
//            e2[0]=edgelist2[j].p1.x;e2[1]=edgelist2[j].p1.y;
//            e2[2]=edgelist2[j].p2.x;e2[3]=edgelist2[j].p2.y;
//
//
//            if (e1[0]==e2[0]&&e1[1]==e2[1])
//            {
//                outP[3].x=e1[0];
//                outP[3].y=e1[1];
//
//            }
//            if (e1[0]==e2[2]&&e1[1]==e2[3])
//            {
//                outP[3].x=e1[0];
//                outP[3].y=e1[1];
//
//            }
//            if (e1[2]==e2[0]&&e1[3]==e2[1])
//            {
//                outP[3].x=e1[2];
//                outP[3].y=e1[3];
//
//            }
//            if (e1[2]==e2[2]&&e1[3]==e2[3])
//            {
//                outP[3].x=e1[2];
//                outP[3].y=e1[3];
//
//            }
//
//        }
//
//    }
//    for (int i = 0; i <p3EdgeIndex.size() ; ++i)
//    {
//        for (int j = 0; j < p5EdgeIndex.size(); ++j)
//        {
//            cv::Vec4f e1,e2;
//            e1[0]=edgelist2[i].p1.x;e1[1]=edgelist2[i].p1.y;
//            e1[2]=edgelist2[i].p2.x;e1[3]=edgelist2[i].p2.y;
//
//            e2[0]=edgelist2[j].p1.x;e2[1]=edgelist2[j].p1.y;
//            e2[2]=edgelist2[j].p2.x;e2[3]=edgelist2[j].p2.y;
//
//
//            if (e1[0]==e2[0]&&e1[1]==e2[1])
//            {
//                outP[4].x=e1[0];
//                outP[4].y=e1[1];
//
//            }
//            if (e1[0]==e2[2]&&e1[1]==e2[3])
//            {
//                outP[4].x=e1[0];
//                outP[4].y=e1[1];
//
//            }
//            if (e1[2]==e2[0]&&e1[3]==e2[1])
//            {
//                outP[4].x=e1[2];
//                outP[4].y=e1[3];
//
//            }
//            if (e1[2]==e2[2]&&e1[3]==e2[3])
//            {
//                outP[4].x=e1[2];
//                outP[4].y=e1[3];
//
//            }
//
//        }
//
//    }
//    for (int i = 0; i <p4EdgeIndex.size() ; ++i)
//    {
//        for (int j = 0; j < p5EdgeIndex.size(); ++j)
//        {
//            cv::Vec4f e1,e2;
//            e1[0]=edgelist2[i].p1.x;e1[1]=edgelist2[i].p1.y;
//            e1[2]=edgelist2[i].p2.x;e1[3]=edgelist2[i].p2.y;
//
//            e2[0]=edgelist2[j].p1.x;e2[1]=edgelist2[j].p1.y;
//            e2[2]=edgelist2[j].p2.x;e2[3]=edgelist2[j].p2.y;
//
//
//            if (e1[0]==e2[0]&&e1[1]==e2[1])
//            {
//                outP[5].x=e1[0];
//                outP[5].y=e1[1];
//
//            }
//            if (e1[0]==e2[2]&&e1[1]==e2[3])
//            {
//                outP[5].x=e1[0];
//                outP[5].y=e1[1];
//
//            }
//            if (e1[2]==e2[0]&&e1[3]==e2[1])
//            {
//                outP[5].x=e1[2];
//                outP[5].y=e1[3];
//
//            }
//            if (e1[2]==e2[2]&&e1[3]==e2[3])
//            {
//                outP[5].x=e1[2];
//                outP[5].y=e1[3];
//
//            }
//
//        }
//
//    }
//
//   //这里得到三个点
//    Vertex<float> _p0(p0.x,p0.y);
//    Vertex<float> _p1(p1.x,p1.y);
//    Vertex<float> _p2(outP[0].x,outP[0].y);
//    Triangle<float> outT1(_p0,_p1,_p2);
//    Vertex<float> _p3(p3.x,p3.y);
//    Vertex<float> _p4(p4.x,p4.y);
//    Vertex<float> _p5(outP[3].x,outP[3].y);
//    Triangle<float> _outT1(_p3,_p4,_p5);
//
//    Vertex<float> p_0(p0.x,p0.y);
//    Vertex<float> p_1(p2.x,p2.y);
//    Vertex<float> p_2(outP[1].x,outP[1].y);
//    Triangle<float> outT_1(p_0,p_1,p_2);
//    Vertex<float> _p_0(p3.x,p3.y);
//    Vertex<float> _p_1(p5.x,p5.y);
//    Vertex<float> _p_2(outP[4].x,outP[4].y);
//    Triangle<float> outT_2(_p_0,_p_1,_p_2);
//
//    Vertex<float> p0_(p1.x,p1.y);
//    Vertex<float> p1_(p2.x,p2.y);
//    Vertex<float> p2_(outP[2].x,outP[2].y);
//    Triangle<float> outT_3(p0_,p1_,p2_);
//    Vertex<float> p2_1(p4.x,p4.y);
//    Vertex<float> p2_2(p5.x,p5.y);
//    Vertex<float> p2_3(outP[5].x,outP[5].y);
//    Triangle<float> outT_4(p2_1,p2_2,p2_3);
//
//    double similrity1=SimilarityValue(outT1,_outT1);
//    double similrity2=SimilarityValue(outT_1,outT_2);
//    double similrity3=SimilarityValue(outT_3,outT_4);
//    //cout<<"similrity1:"<<similrity1<<endl;
//
//    if(similrity1<0.7||similrity2<0.7||similrity3<0.7)
//    {
//        return 0;
//    }
//
//    return 1;
//
//}
//
//double localtion_constriant(Triangle<float> t1,Triangle<float>t2)
//{
//    Vertex<float> vo(t1.p1.x-t1.circum.x,t1.p1.y-t1.circum.y);
//    Vertex<float> vv1(t1.p2.x-t1.p1.x,t1.p2.y-t1.p1.y);
//    double ori1=vo.x*vv1.x+vo.y*vv1.y;
//    Vertex<float> vv2(t1.p3.x-t1.p1.x,t1.p3.y-t1.p1.y);
//    double ori2=(vo.x*vv2.x+vo.y*vv2.y);
//
//
//
//    Vertex<float> _vo(t2.p1.x-t2.circum.x,t2.p1.y-t2.circum.y);
//    Vertex<float> _vv1(t2.p2.x-t2.p1.x,t2.p2.y-t2.p1.y);
//    double _ori1=_vo.x*_vv1.x+_vo.y*_vv1.y;
//    Vertex<float> _vv2(t2.p3.x-t2.p1.x,t1.p3.y-t2.p1.y);
//    double _ori2=_vo.x*_vv1.x+_vo.y*_vv1.y;
//
//    if(ori1)
//
//return 0;
//
//}
//*/
void get_circle_coordinates(Mat img,Triangle<float> t1) {

    Mat roi = Mat::zeros(img.size(), CV_8U);
    Mat dst;
    vector<vector<Point>> contour;
    vector<Point> pts;
    pts.push_back(Point(t1.p1.x, t1.p1.y));
    pts.push_back(Point(t1.p2.x, t1.p2.y));
    pts.push_back(Point(t1.p3.x, t1.p3.y));
    contour.push_back(pts);
    drawContours(roi, contour, 0, Scalar::all(255), -1);
    img.copyTo(dst, roi);
    //imshow("roi",roi);
    //imshow("img",img);
    //imshow("dst",dst);

//计算外矩阵
    vector<Rect> boundRect(contour.size());//定义外接矩阵合集
    vector<RotatedRect> box(contour.size());//定义最小外接矩阵集合
    Point2f rect[4];
    for (int i = 0; i < contour.size(); ++i) {
        box[i] = minAreaRect(contour[i]); //计算每个轮廓最小外接矩形
        boundRect[i] = boundingRect(Mat(contour[i]));
        // circle(img,Point(box[i].center.x,box[i].center.y),5,Scalar(0,255,0),-1,8);
        box[i].points(rect);//把最小外接矩形四个端点复制给rect数组
        //rectangle(img, Point(boundRect[i].x, boundRect[i].y),
        // Point(boundRect[i].x + boundRect[i].width,   boundRect[i].y + boundRect[i].height),Scalar(0, 255, 0), 2, 8);



        //左上角
        // circle(img,Point(boundRect[i].x,boundRect[i].y),5,Scalar(0,0,255),-1,8);
        //右上
        //circle(img, Point(boundRect[i].x+boundRect[i].width,boundRect[i].y),5,Scalar(0,0,255),-1,8);
        //左下
        // circle(img,Point(boundRect[i].x,boundRect[i].y+boundRect[i].height),5,Scalar(0,0,255),-1,8);
        //右下角
        // circle(img,Point(boundRect[i].x+boundRect[i].width, boundRect[i].y+boundRect[i].height),5,Scalar(0,0,255),-1,8);


        //最小矩形
//        for (int j = 0; j <4 ; ++j)
//        {
//            line(img,rect[j],rect[(j+1)%4],Scalar(0,0,255),2,8);
//
//        }
        //imshow("img", img);
        //显示矩形局部区域图像
        Mat _img;
        img.copyTo(_img);
        cv::Mat roiImage = _img(Rect(boundRect[i].x, boundRect[i].y, boundRect[i].width, boundRect[i].height));
        imshow("roiImage", roiImage);
        waitKey(0);


        //提取像素点
        /*  vector<int>vec;
          for(int i = 0; i < roiImage.rows; i++)//rows->height->Y
          {
              for (int j = 0; j < roiImage.cols; j++)//cols->width->X//提取像素值，按照i为rows, j为cols。
              {


                  if (roiImage.at<uchar>(i, j) != 0)
                  {
                      vec.push_back(roiImage.at<uchar>(i, j));


                  }


              }


          }
          for (auto i = vec.begin(); i <vec.end() ; ++i) {
              cout<<*i<<" ";

          }


      }*/
    }
}
    double Entropy(Mat img)
    {
        double temp[256] = {0.0};

        // 计算每个像素的累积值
        for (int m = 0; m < img.rows; m++) {// 有效访问行列的方式
            const uchar *t = img.ptr<uchar>(m);
            for (int n = 0; n < img.cols; n++) {
                int i = t[n];
                temp[i] = temp[i] + 1;
            }
        }

        // 计算每个像素的概率
        for (int i = 0; i < 256; i++) {
            temp[i] = temp[i] / (img.rows * img.cols);
        }

        double result = 0;
        // 计算图像信息熵
        for (int i = 0; i < 256; i++) {
            if (temp[i] == 0.0)
                result = result;
            else
                result = result - temp[i] * (log(temp[i]) / log(2.0));
        }

        return result;

    }

// 两幅图像联合信息熵计算

    double ComEntropy(Mat img1, Mat img2, double img1_entropy, double img2_entropy)
    {
        double temp[256][256] = {0.0};

        // 计算联合图像像素的累积值
        for (int m1 = 0, m2 = 0; m1 < img1.rows, m2 < img2.rows; m1++, m2++) {    // 有效访问行列的方式
            const uchar *t1 = img1.ptr<uchar>(m1);
            const uchar *t2 = img2.ptr<uchar>(m2);
            for (int n1 = 0, n2 = 0; n1 < img1.cols, n2 < img2.cols; n1++, n2++) {
                int i = t1[n1], j = t2[n2];
                temp[i][j] = temp[i][j] + 1;
            }
        }

        // 计算每个联合像素的概率
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                temp[i][j] = temp[i][j] / (img1.rows * img1.cols);
            }
        }

        double result = 0.0;
        //计算图像联合信息熵
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                if (temp[i][j] == 0.0)
                    result = result;
                else
                    result = result - temp[i][j] * (log(temp[i][j]) / log(2.0));
            }
        }

        //得到两幅图像的互信息熵
        img1_entropy = Entropy(img1);
        img2_entropy = Entropy(img2);
        result = img1_entropy + img2_entropy - result;

        return result;

    }

