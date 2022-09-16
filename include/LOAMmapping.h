#include"utility.h"



class LOAMmapping : public ParamServer
{
    private: 
        float transformTobeMapped[6];
        float transformGuess[6];
    public: 
        Eigen::Affine3f affine_out;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
        pcl::PointCloud<PointType>::Ptr lidarCloudCornerLastDS;
        pcl::PointCloud<PointType>::Ptr lidarCloudSurfLastDS;
        pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMapDS;
        pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMapDS;

        pcl::PointCloud<PointType>::Ptr lidarCloudOri;
        pcl::PointCloud<PointType>::Ptr coeffSel;

        std::vector<PointType> lidarCloudOriCornerVec; // corner point holder for parallel computation
        std::vector<PointType> coeffSelCornerVec;
        std::vector<bool> lidarCloudOriCornerFlag;
        std::vector<PointType> lidarCloudOriSurfVec; // surf point holder for parallel computation
        std::vector<PointType> coeffSelSurfVec;
        std::vector<bool> lidarCloudOriSurfFlag;
        vector<double> mapRegistrationError;

        int iterCount = 0;
        int lidarCloudCornerLastDSNum = 0;
        int lidarCloudSurfLastDSNum = 0;

        int edgePointCorrNum = 0;
        int surfPointCorrNum = 0;

        double inlier_ratio = 0;
        double inlier_ratio2 = 0;
        float cornerTime = 0, surfTime = 0, optTime = 0;
        double regiError;

        double minEigen = 1e+6;

        bool isDegenerate = false;
        cv::Mat matP;

        LOAMmapping(pcl::PointCloud<PointType>::Ptr lidarCloudCornerLastDSnew,pcl::PointCloud<PointType>::Ptr lidarCloudSurfLastDSnew,
            pcl::PointCloud<PointType>::Ptr lidarCloudCornerFromMapDSnew, pcl::PointCloud<PointType>::Ptr lidarCloudSurfFromMapDSnew,
            const Eigen::Affine3f affine_guess_new) // you don't wanna change affine_guess_new here
        {
            kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
            kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());
            lidarCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
            lidarCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
            lidarCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
            lidarCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*lidarCloudCornerFromMapDSnew,*lidarCloudCornerFromMapDS);
            pcl::copyPointCloud(*lidarCloudSurfFromMapDSnew,*lidarCloudSurfFromMapDS);
            kdtreeCornerFromMap->setInputCloud(lidarCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(lidarCloudSurfFromMapDS);
            pcl::copyPointCloud(*lidarCloudCornerLastDSnew,*lidarCloudCornerLastDS);
            pcl::copyPointCloud(*lidarCloudSurfLastDSnew,*lidarCloudSurfLastDS);

            lidarCloudCornerLastDSNum = lidarCloudCornerLastDS->points.size();
            lidarCloudSurfLastDSNum = lidarCloudSurfLastDS->points.size();
            affine_out = affine_guess_new;

            Affine3f2Trans(affine_out, transformTobeMapped);
            Affine3f2Trans(affine_out, transformGuess);

            lidarCloudOri.reset(new pcl::PointCloud<PointType>());
            coeffSel.reset(new pcl::PointCloud<PointType>());

            lidarCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
            coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
            lidarCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
            lidarCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
            coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
            lidarCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

            std::fill(lidarCloudOriCornerFlag.begin(), lidarCloudOriCornerFlag.end(), false);
            std::fill(lidarCloudOriSurfFlag.begin(), lidarCloudOriSurfFlag.end(), false);
            matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
        }

        void match( )
        {
            iterCount = 0;
            for (; iterCount < optIteration; iterCount++)
            {
                lidarCloudOri->clear();
                coeffSel->clear();
                TicToc corner;
                cornerOptimization(iterCount);
                cornerTime += corner.toc();
                TicToc surf;
                surfOptimization(iterCount);
                surfTime += surf.toc();
                TicToc opt;
                combineOptimizationCoeffs();
                optTime += opt.toc();
                inlier_ratio2 = (double)(surfPointCorrNum + edgePointCorrNum)/( lidarCloudSurfLastDSNum+ lidarCloudCornerLastDSNum);
                // surf only
                // inlier_ratio2 = (double)surfPointCorrNum/ lidarCloudSurfLastDSNum;

                // it is actually for map extension, otherwise cv error happens
                // 0.2 or higher would seriously deteriarate the performance
                if (inlier_ratio2 < 0.1 ) break; 
                if (LMOptimization(iterCount) == true)
                {
                    // cout<<"converged"<<endl;
                    break;              
                }
            }
            // cout<<cornerTime<<" "<<surfTime<<" edge corr: "<<edgePointCorrNum <<" surf corr: "<<surfPointCorrNum<<endl; 
            // lidarCloudSelNum = errorSize
            
            if (mapRegistrationError.empty()) 
            {
                inlier_ratio = 0;
                minEigen = 0;
                regiError = 0;
                return;
            }
            regiError = accumulate(mapRegistrationError.begin(),mapRegistrationError.end(),0.0);
            double errorSize = mapRegistrationError.size();
            regiError /= errorSize;                        
            int inlierCnt = 0;
            for (int i = 0; i < errorSize; i++)
            {
                // regiError2 += mapRegistrationError[i]*mapRegistrationError[i];
                if (mapRegistrationError[i] < inlierThreshold) inlierCnt++; // <0.1 means "properly matched"
            }
            // regiError2 = sqrt(regiError2/errorSize);
            inlier_ratio = inlierCnt/ errorSize;
            // ROS_INFO_STREAM(" mapping error: "<<regiError);
        }


    void cornerOptimization(int iterCount)
    {
        affine_out = trans2Affine3f(transformTobeMapped);

        // #pragma omp parallel for num_threads(numberOfCores) // runtime error, don't use it!
        for (int i = 0; i < lidarCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = lidarCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += lidarCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += lidarCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += lidarCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = lidarCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = lidarCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = lidarCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) 
                    {
                        lidarCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        lidarCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount)
    {
        affine_out = trans2Affine3f(transformTobeMapped);

        for (int i = 0; i < lidarCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = lidarCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) 
                {
                    matA0(j, 0) = lidarCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = lidarCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = lidarCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                // why Ax = B, means x is the unit normal vector of the plane?
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;
 
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * lidarCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * lidarCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * lidarCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;
                    if (s > 0.1) {
                        lidarCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        lidarCloudOriSurfFlag[i] = true;   
                    }
                }

            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        edgePointCorrNum = 0;
        surfPointCorrNum = 0;
        for (int i = 0; i < lidarCloudCornerLastDSNum; ++i){
            if (lidarCloudOriCornerFlag[i] == true){
                lidarCloudOri->push_back(lidarCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
                edgePointCorrNum++;
            }
        }
        // combine surf coeffs
        for (int i = 0; i < lidarCloudSurfLastDSNum; ++i){
            if (lidarCloudOriSurfFlag[i] == true){
                lidarCloudOri->push_back(lidarCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
                surfPointCorrNum++;
            }
        }
        // reset flag for next iteration
        std::fill(lidarCloudOriCornerFlag.begin(), lidarCloudOriCornerFlag.end(), false);
        std::fill(lidarCloudOriSurfFlag.begin(), lidarCloudOriSurfFlag.end(), false);


    }

    bool LMOptimization(int iterCount)
    {
        // float s1 = sin(transformTobeMapped[0]);
        // float c1 = cos(transformTobeMapped[0]);
        // float s2 = sin(transformTobeMapped[1]);
        // float c2 = cos(transformTobeMapped[1]);
        // float s3 = sin(transformTobeMapped[2]);
        // float c3 = cos(transformTobeMapped[2]);

        // int lidarCloudSelNum = lidarCloudOri->size();
        // if (lidarCloudSelNum < 50) {
        //     return false;
        // }

        // cv::Mat matA(lidarCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        // cv::Mat matAt(6, lidarCloudSelNum, CV_32F, cv::Scalar::all(0));
        // cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        // cv::Mat matB(lidarCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        // cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        // cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        // PointType pointOri, coeff;
        // notice  the subscript in wiki is the sequence, not x y z!!!
        // for (int i = 0; i < lidarCloudSelNum; i++) {
        //     pointOri.x = lidarCloudOri->points[i].x;
        //     pointOri.y = lidarCloudOri->points[i].y;
        //     pointOri.z = lidarCloudOri->points[i].z;
        //     coeff.x = coeffSel->points[i].x;
        //     coeff.y = coeffSel->points[i].y;
        //     coeff.z = coeffSel->points[i].z;
        //     coeff.intensity = coeffSel->points[i].intensity;
        //     // type it right the 3rd time. but why so slow?
        //     float arx = ((-s1*s3+c1*s2*c3)*pointOri.x + (-s1*c3-c1*s2*s3)*pointOri.y - c1*c2*pointOri.z) * coeff.y
        //               + ((c1*s3+s1*s2*c3)*pointOri.x + (c1*c3-s1*s2*s3)*pointOri.y - s1*c2*pointOri.z) * coeff.z;

        //     float ary = (-s2*c3*pointOri.x + s2*s3*pointOri.y + c2*pointOri.z) * coeff.x
        //               + (s1*c2*c3*pointOri.x - s1*c2*s3*pointOri.y + s1*s2*pointOri.z) * coeff.y
        //               + (-c1*c2*c3*pointOri.x + c1*c2*s3*pointOri.y - c1*s2*pointOri.z) * coeff.z;

        //     float arz = (-c2*s3*pointOri.x - c2*c3*pointOri.y)*coeff.x
        //               + ((c1*c3-s1*s2*s3)*pointOri.x + (-c1*s3-s1*s2*c3)*pointOri.y) * coeff.y
        //               + ((s1*c3 + c1*s2*s3)*pointOri.x + (-s1*s3+c1*s2*c3)*pointOri.y)*coeff.z;
        //     matA.at<float>(i, 0) = arx;
        //     matA.at<float>(i, 1) = ary;
        //     matA.at<float>(i, 2) = arz;
        //     matA.at<float>(i, 3) = coeff.x;
        //     matA.at<float>(i, 4) = coeff.y;
        //     matA.at<float>(i, 5) = coeff.z;
        //     matB.at<float>(i, 0) = -coeff.intensity; // -f, no 0.05 here,same in LOAM
        // }

        // this way much faster, why?
       // lidar -> camera

        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int lidarCloudSelNum = lidarCloudOri->size();

        cv::Mat matA(lidarCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, lidarCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(lidarCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;
        mapRegistrationError.clear(); // only get the last iteration error
        for (int i = 0; i < lidarCloudSelNum; i++) {
            mapRegistrationError.push_back(fabs(coeffSel->points[i].intensity));
            // lidar -> camera
            pointOri.x = lidarCloudOri->points[i].y;
            pointOri.y = lidarCloudOri->points[i].z;
            pointOri.z = lidarCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // camera -> lidar
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) 
        {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matVf(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matVu(6, 6, CV_32F, cv::Scalar::all(0));
            cv::eigen(matAtA, matE, matVf);
            matVf.copyTo(matVu);

            isDegenerate = false;
            float eigenThre[6] = {0,0,0,0,0,0}; // 100 may not be good
            float degeneracyThre =100;
            for (int i = 0; i < 6; i++) eigenThre[i] = degeneracyThre;
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < minEigen) minEigen = matE.at<float>(0, i);
                if (matE.at<float>(0, i) < eigenThre[i]) 
                {
                    for (int j = 0; j < 6; j++) 
                    {
                        matVu.at<float>(i, j) = 0; // zero out the underconstrained DOFs, which will not be updated on transformTobeMapped
                    }
                    isDegenerate = true;
                }
                else {
                    break;
                }
            }
            matP = matVf.inv() * matVu;
        }
        
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2; // here matX is xu', matX2 is xu
        }
        
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }



    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = affine_out(0,0) * pi->x + affine_out(0,1) * pi->y + affine_out(0,2) * pi->z + affine_out(0,3);
        po->y = affine_out(1,0) * pi->x + affine_out(1,1) * pi->y + affine_out(1,2) * pi->z + affine_out(1,3);
        po->z = affine_out(2,0) * pi->x + affine_out(2,1) * pi->y + affine_out(2,2) * pi->z + affine_out(2,3);
        po->intensity = pi->intensity;
    }

    void getTransformation(float transformTobeMappedI[6])
    {
        for (int i = 0; i < 6; i++) transformTobeMappedI[i] = transformTobeMapped[i];
    }

};