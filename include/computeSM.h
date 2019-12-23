//
// Created by lu on 19-4-17.
//

#ifndef ORB_DTM_COMPUTESM_H
#define ORB_DTM_COMPUTESM_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "Vertex.h"
#include "edge.h"
#include "triangle.h"
#include "delaunay.h"



// 计算相似度
double SimilarityValue(Triangle<float> t1,Triangle<float>t2);

// 计算相似度矩阵
void ComputeSimilarityMatrix(const Delaunay<float> & net1, const Delaunay<float> & net2, Eigen::MatrixXd &similarityMatrix);

double checkconstriant(const Delaunay<float> & net1, const Delaunay<float> & net2,const int t1Index,const int t2Index);

void get_circle_coordinates(Mat img,Triangle<float> t1);

double ComEntropy(Mat img1, Mat img2, double img1_entropy, double img2_entropy);
#endif //ORB_DTM_COMPUTESM_H
