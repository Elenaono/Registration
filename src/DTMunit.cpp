//
// Created by lu on 19-5-11.
//
#include "DTMunit.h"
#include"dataimg.h"
#include "computeSM.h"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include"opencv2/core/core.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
/**
 * @brief 构建DTM的基本函数
 * @param initGood_matches
 * @param mvKeys1
 * @param mvKeys2
 * @param feature1
 * @param feature2
 * @return newGood_matches
 */
 using namespace std;
 using namespace cv;


vector<DMatch> ComputeDTMunit(int threshold, const vector<DMatch> &initGood_matches, const vector<cv::KeyPoint> &mvKeys1, const vector<cv::KeyPoint> &mvKeys2, cv::Mat &feature1, cv::Mat &feature2 ) {
    if (initGood_matches.empty())
        return initGood_matches;
    Mat feature3 = feature1.clone();
    Mat feature4 = feature2.clone();
    ///delaunay one
//    cout << "DT one:" << endl;
    vector<Vertex<float> > points1;//二维点坐标集合
    for (const auto &p:initGood_matches) {
        points1.emplace_back(Vertex<float>(mvKeys1[p.queryIdx].pt.x, mvKeys1[p.queryIdx].pt.y, p.imgIdx));
    }

    Delaunay<float> triangulation1;
    const std::vector<Triangle<float> > triangles1 = triangulation1.Triangulate(points1);  //逐点插入法
    triangulation1.ComputeEdgeMatrix();
    std::cout << "三角形初始个数" << triangles1.size() << " triangles generated" << endl;
    const std::vector<Edge<float> > edges1 = triangulation1.GetEdges();
    const std::vector<Triangle<float> > tri1 = triangulation1.GetTriangles();

    for (const auto &e : edges1) {
        line(feature1, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }
    Mat mask;
    img1.copyTo(mask);
    for (const auto &t : triangles1) {
        double sideLength;
        sideLength = sqrt((t.mainpoint.x - t.circum.x) * (t.mainpoint.x - t.circum.x) +
                          (t.mainpoint.y - t.circum.y) * (t.mainpoint.y - t.circum.y));
//        cout << "sidelength: " << sideLength << endl;
        //cout<<"坐标"<<t.circum.x<<","<<t.circum.y<<endl;
        if (sideLength < MAX_ARROR_SIZE) {
            //circle(feature1, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
            //arrowedLine(feature1, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),1, 8);

            int radius = t.circum_radius;
            Point p(t.circum.x, t.circum.y);
            //cout<<"radius"<<radius<<endl;




        }
    }


    ///delaunay two
//    cout << "DT two:" << endl;
    vector<Vertex<float> > points2;
    for (const auto &p:initGood_matches) {
        points2.emplace_back(Vertex<float>(mvKeys2[p.trainIdx].pt.x, mvKeys2[p.trainIdx].pt.y, p.imgIdx));
    }

    Delaunay<float> triangulation2;
    const std::vector<Triangle<float> > triangles2 = triangulation2.Triangulate(points2);  //逐点插入法
    triangulation2.ComputeEdgeMatrix();
//    std::cout << "\t\t" <<triangles2.size() << " triangles generated"<<endl;
    const std::vector<Edge<float> > edges2 = triangulation2.GetEdges();
    const std::vector<Triangle<float> > tri2 = triangulation2.GetTriangles();

    for (const auto &e : edges2) {
        line(feature2, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }
    for (const auto &t : triangles2) {
        double sideLength;
        sideLength = sqrt((t.mainpoint.x - t.circum.x) * (t.mainpoint.x - t.circum.x) +
                          (t.mainpoint.y - t.circum.y) * (t.mainpoint.y - t.circum.y));
//        cout << "sidelength: " << sideLength << endl;
        if (sideLength < MAX_ARROR_SIZE) {
            //circle(feature2, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
            //arrowedLine(feature2, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),1, 8);
        }
    }


    /**************** 显示匹配结果与初始DT网络 ******************/
//    cout << "\t匹配:" << endl;
    cout << "\t\tInitialmatch:" << initGood_matches.size() << endl;
    Mat beforeOpt;
    Mat Init1, Init2;
    img1.copyTo(Init1);
    img2.copyTo(Init2);
    cv::drawMatches(Init1, mvKeys1, Init2, mvKeys2, initGood_matches, beforeOpt);
    imshow("before optimization", beforeOpt);
    imwrite("./figure/beforeDTM.png", beforeOpt);
    waitKey(0);

/*******************  构建边矩阵，并计算相似度(范数)，进行DT网络的优化  *********************/
//    cout << "\n计算DTM的相关信息：" << endl;
    Eigen::MatrixXd::Index maxRow, maxCol;
    Eigen::MatrixXd edgeMatrix = Eigen::MatrixXd::Zero(sizeofEdgeMatrix,
                                                       sizeofEdgeMatrix);  //ComputeEdgeMatrix() 在此处也修改了 20,20 ，需要同步修改，后期改进此处
    edgeMatrix = triangulation1.GetEdgeMatrix() - triangulation2.GetEdgeMatrix();
    //    double value =0;
    //    value = edgeMatrix_.norm();
    //    cout << "\tvalue: " << value <<  endl;      // 相似度

    edgeMatrix.cwiseAbs().colwise().sum().maxCoeff(&maxRow, &maxCol);    // 边矩阵.绝对值.列.和.最大值(行序号,列序号)

//    cout << "提取候选外点：\t"  << maxCol << endl;
//    cout << "显示sum:\n" << edgeMatrix_.cwiseAbs().colwise().sum() << endl;
//    cout << "计算列和：\n" << edgeMatrix_.cwiseAbs().colwise().sum()<< endl;
//    cout << "显示边矩阵之差：\n"<< edgeMatrix_.cwiseAbs().col(maxCol).transpose() << endl;
//    cout << "二者之差：\n" << edgeMatrix_.cwiseAbs().colwise().sum() - edgeMatrix_.cwiseAbs().col(maxCol).transpose()<< endl;
//    cout << "候选外点：" << mvKeys2[good_matches[maxCol].trainIdx].pt << endl;

    // 通过DT网络的边矩阵之差的范数，删除列和较大的候选外点集
    vector<DMatch> newGood_matches(initGood_matches);
//    cout << "\nold size:\t" << newGood_matches.size()<<endl;
    for (int i = sizeofEdgeMatrix; i != 0; i--) {
        if ((edgeMatrix.cwiseAbs().colwise().sum())(0, i - 1) >= threshold) {
//            cout << (edgeMatrix_.cwiseAbs().colwise().sum())(0,i-1) << "\t,\t" << mvKeys1[newGood_matches[i-1].queryIdx].pt <<"\t,\t" << mvKeys2[newGood_matches[i-1].trainIdx].pt << endl;
            newGood_matches.erase(newGood_matches.begin() + i - 1);
        }
    }
//    cout << "new size:\t" << newGood_matches.size()<<endl;

    /************ 显示优化后的DT网络 ****************/
    if (newGood_matches.empty())
        return newGood_matches;
    ///delaunay three
    std::vector<Vertex<float> > points3;
    for (const auto &g:newGood_matches) {
        points3.emplace_back(Vertex<float>(mvKeys1[g.queryIdx].pt.x, mvKeys1[g.queryIdx].pt.y, g.imgIdx));
    }
    Delaunay<float> triangulation3;
    const std::vector<Triangle<float> > triangles3 = triangulation3.Triangulate(points3);  //逐点插入法
    triangulation3.ComputeEdgeMatrix();
    std::cout << "迭代后三角形个数：" << triangles3.size() << " triangles generated" << endl;
    const std::vector<Edge<float> > edges3 = triangulation3.GetEdges();
    for (const auto &e : edges3) {
        line(feature3, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }
    for (const auto &t : triangles3) {
        double sideLength;
        sideLength = sqrt((t.mainpoint.x - t.circum.x) * (t.mainpoint.x - t.circum.x) +
                          (t.mainpoint.y - t.circum.y) * (t.mainpoint.y - t.circum.y));
//        cout << "sidelength: " << sideLength << endl;
        if (sideLength < MAX_ARROR_SIZE) {
            circle(feature3, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
            arrowedLine(feature3, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),
                        1, 8);
        }
    }

    ///delaunay four
//    cout << "\tDT four:" << endl;
    std::vector<Vertex<float> > points4;
    for (const auto &g:newGood_matches) {
        points4.emplace_back(Vertex<float>(mvKeys2[g.trainIdx].pt.x, mvKeys2[g.trainIdx].pt.y, g.imgIdx));
    }

    Delaunay<float> triangulation4;
    const std::vector<Triangle<float> > triangles4 = triangulation4.Triangulate(points4);  //逐点插入法
    triangulation4.ComputeEdgeMatrix();
//    std::cout << "\t\t" << triangles4.size() << " triangles generated"<<endl;
    const std::vector<Edge<float> > edges4 = triangulation4.GetEdges();

    for (const auto &e : edges4) {
        line(feature4, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }
    for (const auto &t : triangles4) {
        double sideLength;
        sideLength = sqrt((t.mainpoint.x - t.circum.x) * (t.mainpoint.x - t.circum.x) +
                          (t.mainpoint.y - t.circum.y) * (t.mainpoint.y - t.circum.y));
//        cout << "sidelength: " << sideLength << endl;
        if (sideLength < MAX_ARROR_SIZE) {
            circle(feature4, Point(t.circum.x, t.circum.y), 0.1, Scalar(0, 255, 0));
            arrowedLine(feature4, Point(t.circum.x, t.circum.y), Point(t.mainpoint.x, t.mainpoint.y), Scalar(0, 255, 0),
                        1, 8);
        }
    }


    Mat afterOpt;
    cv::drawMatches(feature3,mvKeys1,feature4,mvKeys2,newGood_matches,afterOpt);
    cout<<"goodmatches:"<<newGood_matches.size()<<endl;
    imshow("after optimization",afterOpt);
    imwrite("./figure/DTM.png",afterOpt);
    waitKey(0);

//    cout << "Finished in function!!!" << endl;
   // return newGood_matches;


    //三角形相似

    Eigen::MatrixXd similarityMatrix(3000,3000);
    ComputeSimilarityMatrix(triangulation3,triangulation4,similarityMatrix);
    vector<Point2f>pf1;
    vector<Point2f>pf2;
    int count=0;
    for (int i = 0; i <triangles1.size() ; i++) {
        for (int j = 0; j <triangles2.size() ; j++) {
            if ((similarityMatrix(i,j))!=0 && checkconstriant(triangulation3,triangulation4,i,j))
            {
                Point2f  p1,p2,p3;

                p1.x=triangles1[i].p1.x;
                p1.y=triangles1[i].p1.y;
                p2.x=triangles1[i].p2.x;
                p2.y=triangles1[i].p2.y;
                p3.x=triangles1[i].p3.x;
                p3.y=triangles1[i].p3.y;
                Rect rect(0,0,img1.size().width,img1.size().height);
                if (rect.contains(p1)&&rect.contains(p2)&&rect.contains(p3))
                {
                    pf1.push_back(p1);
                    pf1.push_back(p2);
                    pf1.push_back(p3);
                    count++;
                }

                Point2f  p4,p5,p6;

                p4.x=triangles2[j].p1.x;
                p4.y=triangles2[j].p1.y;
                p5.x=triangles2[j].p2.x;
                p5.y=triangles2[j].p2.y;
                p6.x=triangles2[j].p3.x;
                p6.y=triangles2[j].p3.y;

                Rect _rect(0,0,img2.size().width,img2.size().height);
                if (_rect.contains(p4)&&_rect.contains(p5)&&_rect.contains(p6))
                {
                    pf2.push_back(p4);
                    pf2.push_back(p5);
                    pf2.push_back(p6);
                }

            }

        }

    }
   // cout<<"origin_pf1:"<<pf1.size()<<endl;
    //cout<<"origin_pf2:"<<pf2.size()<<endl;

    //去除重复的点
    vector<Point2f>::iterator it1,it;
    for (it=pf1.begin()+1; it!=pf1.end(); )
    {
        it1=find(pf1.begin(),it,*it);

        if(it1!=it)
        {
            it=pf1.erase(it);
        } else{
            it++;
        }

    }


    vector<Point2f>::iterator it2,its;
    for (its=pf2.begin()+1; its!=pf2.end(); )
    {
        it2=find(pf2.begin(),its,*its);

        if(it2!=its)
        {
            its=pf2.erase(its);
        } else{
            its++;
        }

    }

    vector<KeyPoint>kp1,kp2;
    KeyPoint::convert(pf1,kp1,1,1,0,-1);
    KeyPoint::convert(pf2,kp2,1,1,0,-1);
    Mat kp1_img,kp2_img;
    img1.copyTo(kp1_img);
    img2.copyTo(kp2_img);

    drawKeypoints(kp1_img,kp1,kp1_img,Scalar(0,0,255));
    drawKeypoints(kp2_img,kp1,kp2_img,Scalar(0,0,255));
    imshow("triangle1",kp1_img);
    imshow("triangle2",kp2_img);
    waitKey(0);


/*
    vector<Vertex<float > > point_kp1;
    for(int i=0;i<pf1.size();i++)
    {
        point_kp1.emplace_back(Vertex<float>(pf1[i].x, pf1[i].y));

    }

    Delaunay<float> triangulation_kp1;
    const std::vector<Triangle<float> > triangles_kp1 = triangulation_kp1.Triangulate(point_kp1);  //逐点插入法
    cout<<"ssss:"<<triangles_kp1.size()<<endl;
    const std::vector<Edge<float> > edges_kp1 = triangulation_kp1.GetEdges();
    for(const auto &e : edges_kp1)
    {
        line(kp1_img, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }

    /////
    vector<Vertex<float > > point_kp2;

    for(int i=0;i<pf2.size();i++)
    {
        point_kp2.emplace_back(Vertex<float>(pf2[i].x, pf2[i].y));

    }

    Delaunay<float> triangulation_kp2;
    const std::vector<Triangle<float> > triangles_kp2 = triangulation_kp2.Triangulate(point_kp2);  //逐点插入法
    const std::vector<Edge<float> > edges_kp2 = triangulation_kp2.GetEdges();
    for(const auto &e : edges_kp2)
    {
        line(kp2_img, Point(e.p1.x, e.p1.y), Point(e.p2.x, e.p2.y), Scalar(0, 0, 255), 1);
    }

    Mat aftersimilar1,aftersimilar2;
    cout<<"point_kp1:"<<point_kp1.size()<<endl;
    cout<<"point_kp1:"<<point_kp2.size()<<endl;
    imshow("after optimization1",kp1_img);
    imshow("after optimization2",kp2_img);
    imwrite("./figure/DTM1.png",kp1_img);
    imwrite("./figure/DTM2.png",kp2_img);

    waitKey(0);
*/

}



/**
 * @brief 获取剩余点集
 *
 * 输入
 * @param sizeofLevel             剩余点个数
 * @param good_matches
 * @param mvKeys1
 * @param mvKeys2
 * @param mDesc1
 * @param mDesc2
 *
 * 输出
 * @param mvKeys1_new
 * @param mvKeys2_new
 * @param mDes1_new
 * @param mDes2_new
 */
void UpdateKey(const vector<DMatch> &good_matches, const vector<cv::KeyPoint> &mvKeys1, const vector<cv::KeyPoint> &mvKeys2, const cv::Mat &mDes1, const cv::Mat &mDes2,
               vector<cv::KeyPoint> &mvKeys1_new, vector<cv::KeyPoint> &mvKeys2_new, cv::Mat &mDes1_new, cv::Mat &mDes2_new)
{
    //   cv::Mat中没有删除某一列或者行的函数
    //   只能构造新的Mat，在删除某一列后，将后边的复制到新的Mat当中去
    //   新的解决方案是：将Mat转换为vector，使用back() pop()等操作处理后，再转换成Mat
    //   注意：由于删除的是列，而转换成vector后操作的是行，因此可以对Mat进行转置后，再进行转换操作，即Mat.t()
    //   在循环外边完成Mat到vector的转换工作，进行循环操作并退出后，再进行转换回来
    vector<int> order1,order2;
//    cout << "Size of goodmatchs:  " << good_matches.size() << endl;
    // 更新特征点
    for(const auto &g:good_matches)
    {
        order1.emplace_back(g.queryIdx);
        order2.emplace_back(g.trainIdx);
    }
    sort(order1.begin(),order1.end());
    sort(order2.begin(),order2.end());

    // 更新描述子
    int dele_temp_1=0;
    int dele_temp_2=0;
    int dele_temp_count1=0;
    int dele_temp_count2=0;
    for (int i = 0; i < 217; ++i)  //mvKeys1.size()
    {
        if(i == *(order1.begin()+dele_temp_count1))     // 如果与order中的序号相同，则跳过该点
            dele_temp_count1++;
        else
        {
            mvKeys1_new.insert(mvKeys1_new.end(),mvKeys1.begin()+i,mvKeys1.begin()+i+1);
            mDes1.row(i).copyTo(mDes1_new.row(dele_temp_1));
            dele_temp_1++;
        }

        if(i == *(order2.begin()+dele_temp_count2))
            dele_temp_count2++;
        else
        {
            mvKeys2_new.insert(mvKeys2_new.begin()+dele_temp_2,mvKeys2.begin()+i,mvKeys2.begin()+i+1);
            mDes2.row(i).copyTo(mDes2_new.row(dele_temp_2));
            dele_temp_2++;
        }

    }
//    cout << "Sizes of mvKeys1_new: \t" << mvKeys1_new.size() << endl;
//    cout << "Sizes of mDes1_new:\t\t" << mDes1_new.size << endl;
//    cout << "Sizes of mvKeys2_new: \t" << mvKeys2_new.size() << endl;
//    cout << "Sizes of mDes2_new:\t\t" << mDes2_new.size << endl;

}


// sort()时，自定义的排序条件
// 用于对vector对象内的指定成员进行排序
inline bool cmp1(const DMatch first, const DMatch second)
{
    return first.trainIdx < second.trainIdx;
}

// unique()时，自定义的去重条件
// 用于对vector对象内的指定成员进行去重
inline bool cmp2(const DMatch first,const DMatch second)
{
    return first.trainIdx == second.trainIdx;
}
/**
 * @brief 使用BF匹配
 * @param mDes1
 * @param mDes2
 * @return
 */
vector<DMatch> KNNmatchFunc(const cv::Mat &mDes1, const cv::Mat &mDes2)
{
    const float minRatio=1.f/1.2f;
    const int k=2;
    vector<vector<DMatch>> knnmatches;
    vector<DMatch>good_matches;
    FlannBasedMatcher matcher;
    matcher.knnMatch(mDes1,mDes2,knnmatches,k);
    for (std::size_t i = 0; i < knnmatches.size(); ++i)
    {
        const DMatch& bestMatch=knnmatches[i][0];
        const DMatch& betterMatch=knnmatches[i][1];
        float distanceRatio=bestMatch.distance/betterMatch.distance;
        if(distanceRatio<minRatio)
            good_matches.push_back(bestMatch);


    }

    sort(good_matches.begin(), good_matches.end(), cmp1);   //排序
    good_matches.erase(unique(good_matches.begin(),good_matches.end(),cmp2),good_matches.end());    //去重

    // 对新的排列重新赋值index
    for(int i =0 ;i < good_matches.size();i++)
    {
        good_matches[i].imgIdx = i;
    }
    cout<<"暴力匹配后："<<good_matches.size()<<endl;
    return good_matches;


}
vector<DMatch> BFmatchFunc(const cv::Mat &mDes1, const cv::Mat &mDes2, int threshold)
{
//    cout << "\n显示第一次特征匹配的基本信息：" << endl;
    vector<DMatch> matches,good_matches;
    BFMatcher matcher(NORM_L2) ;
    //BFMatcher matcher(NORM_HAMMING) ;

    matcher.match(mDes1,mDes2,matches);

    //计算最大与最小距离
    double min_dist = 10000,max_dist = 0;

    for (int k = 0; k < mDes1.rows; k++)
    {
        double dist = matches[k].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }

//    cout << "\tmin_dist:" << min_dist << endl;
//    cout << "\tmax_dist:" << max_dist << endl;

    //筛选匹配
    int temp=0;
    for (int l = 0; l < mDes1.rows; l++)
    {
        if(matches[l].distance <= 0.6*max_dist)
        {
            matches[l].imgIdx=temp;
            good_matches.emplace_back(matches[l]);
            temp++;
        }
    }
    temp=0;

    sort(good_matches.begin(), good_matches.end(), cmp1);   //排序
    good_matches.erase(unique(good_matches.begin(),good_matches.end(),cmp2),good_matches.end());    //去重

    // 对新的排列重新赋值index
    for(int i =0 ;i < good_matches.size();i++)
    {
        good_matches[i].imgIdx = i;
    }
   cout<<"暴力匹配后："<<good_matches.size()<<endl;
    return good_matches;
}


/**
 * @brief 封装成函数
 *
 * 输入：debugOne,mvKeys1,debugTwo,mvKeys2,control_matches
 * 输出：筛选后的匹配数目
 */
void UsingRansac(const int threshold_value, const cv::Mat &feature1, const cv::Mat &feature2,const vector<cv::KeyPoint> &mvKeys1, const vector<cv::KeyPoint> &mvKeys2,const vector<DMatch> &control_matches)
{
    /***************  RANSAC 实验对照组  ******************************/
//    Mat beforeOpt;
//    cv::drawMatches(feature1,mvKeys1,feature2,mvKeys2,control_matches,beforeOpt);
//    imshow("before optimization",beforeOpt);
//    waitKey(0);

//    保存匹配对序号
    vector<int> queryIdxs( control_matches.size() ), trainIdxs( control_matches.size() );
    for( size_t i = 0; i < control_matches.size(); i++ )
    {
        queryIdxs[i] = control_matches[i].queryIdx;
        trainIdxs[i] = control_matches[i].trainIdx;
    }

    Mat H12;   //变换矩阵

    vector<Point2f> CGpoints1; KeyPoint::convert(mvKeys1, CGpoints1, queryIdxs);
    vector<Point2f> CGpoints2; KeyPoint::convert(mvKeys2, CGpoints2, trainIdxs);
    int ransacReprojThreshold = 5;  //拒绝阈值

    H12 = findHomography( Mat(CGpoints1), Mat(CGpoints2), CV_RANSAC, ransacReprojThreshold );
    vector<char> matchesMask( control_matches.size(), 0 );
    Mat points1t;
    perspectiveTransform(Mat(CGpoints1), points1t, H12);
    int count = 0;
    for( size_t i1 = 0; i1 < CGpoints1.size(); i1++ )  //保存‘内点’
    {
        if( norm(CGpoints2[i1] - points1t.at<Point2f>((int)i1,0)) <= ransacReprojThreshold ) //给内点做标记
        {
            count++;
            matchesMask[i1] = 1;
        }
    }
    cout << "size of control-group matches: " << count << endl;

    Mat afterOpt;   //滤除‘外点’后
    drawMatches(feature1,mvKeys1,feature2,mvKeys2,control_matches,afterOpt,Scalar(0,255,0),Scalar::all(-1),matchesMask);
    imshow("control group",afterOpt);
    imwrite("./figure/RANSAC.png",afterOpt);
    waitKey(0);
//    cout << "Completed in Func!" << endl;
}

