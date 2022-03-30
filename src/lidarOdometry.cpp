// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

// Special notes:
// ALOAM lidar odometry is used since LOAM lidar odometry shows zero drift when static


#include "utility.h"
#include "roll/cloud_info.h"
#include "lidarFactor.hpp"

#define DISTORTION 0
constexpr double SCAN_PERIOD = 0.1;
int corner_correspondence = 0, plane_correspondence = 0;
int opti_num = 2;
// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// // back-up guess for bad opti. results
Eigen::Quaterniond q_backup(1, 0, 0, 0);
Eigen::Vector3d t_backup(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1}; // note the real part position, different from q_w_curr
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

class lidarOdometry : public ParamServer{

private:  

    nav_msgs::Path odometryPath;

    roll::cloud_info cloudInfo;
    std::mutex mtx;

	ros::NodeHandle nh;
    ros::Subscriber subCloudInfo;

    std_msgs::Header cloudHeader;

    ros::Publisher pubLidarOdometry;
    ros::Publisher pubLidarPath;
    ros::Publisher pubCloudInfo;
    bool systemInitedLM;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerCur;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfCur;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerSharp;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFlat;
    int laserCloudCornerSharpNum;
    int laserCloudSurfFlatNum;

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    nav_msgs::Odometry laserOdometry;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    bool isDegenerate;
    cv::Mat matP;

public:
    double regiError;
    double inlierRatio;
    vector<double> odomRegistrationError;
    vector<float> odometryTimeVec;
    vector<vector<double>> odomErrorPerFrame;
    lidarOdometry()
    {
        subCloudInfo = nh.subscribe<roll::cloud_info>("/roll/feature/cloud_info", 1, &lidarOdometry::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        pubLidarOdometry = nh.advertise<nav_msgs::Odometry> ("/roll/lidarOdometry/laser_odom_to_init", 1);      
        pubCloudInfo = nh.advertise<roll::cloud_info> ("/roll/lidarOdometry/cloud_info_with_guess", 1); 
        pubLidarPath = nh.advertise<nav_msgs::Path> ("/roll/lidarOdometry/laser_odom_path", 1); 
        initializationValue();
    }

    void initializationValue()
    {
        odometryPath.poses.clear();
        systemInitedLM = false;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerCur.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfCur.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerSharp.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFlat.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        laserOdometry.header.frame_id =  mapFrame;
        laserOdometry.child_frame_id = lidarFrame;

        laserOdometryTrans.frame_id_ =  mapFrame;
        laserOdometryTrans.child_frame_id_ = lidarFrame;
        
    }

    // undistort lidar point
    void TransformToStart(PointType const *const pi, PointType *const po)
    {
        //interpolation ratio
        double s;
        if (DISTORTION)
            s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
        else
            s = 1.0;
        Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
        Eigen::Vector3d t_point_last = s * t_last_curr;
        Eigen::Vector3d point(pi->x, pi->y, pi->z);
        Eigen::Vector3d un_point = q_point_last * point + t_point_last;
        po->x = un_point.x();
        po->y = un_point.y();
        po->z = un_point.z();
        po->intensity = pi->intensity;
    }
    double rad2deg(double radians)
    {
        return radians * 180.0 / M_PI;
    }

    double deg2rad(double degrees)
    {
        return degrees * M_PI / 180.0;
    }
    
    void optimization()
    {
        TicToc t_opt;
        odomRegistrationError.clear();
        for (int opti_counter = 0; opti_counter < opti_num; ++opti_counter)
        {
            corner_correspondence = 0;
            plane_correspondence = 0;

            //ceres::LossFunction *loss_function = NULL;
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::LocalParameterization *q_parameterization =
                new ceres::EigenQuaternionParameterization();
            ceres::Problem::Options problem_options;

            ceres::Problem problem(problem_options);
            problem.AddParameterBlock(para_q, 4, q_parameterization);
            problem.AddParameterBlock(para_t, 3);

            // corner points
            vector<int> pointSearchInd;
            vector<float> pointSearchSqDis;
            PointType pointSel, coeff, tripod1, tripod2;
            for (int i = 0; i < laserCloudCornerSharpNum; i++)
            {
                TransformToStart(&laserCloudCornerSharp->points[i], &pointSel);
                kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1;
                if (pointSearchSqDis[0] < 25) // as in loam, wtf?
                {
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);

                    float pointSqDis, minPointSqDis2 = 25;
                    for (int j = closestPointInd + 1; j < laserCloudCornerSharpNum; j++)
                    {
                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5*downsampleRate)
                        {
                            break;
                        }

                        pointSqDis = pointDistance(laserCloudCornerLast->points[j], pointSel);

                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan)
                        {
                            if (pointSqDis < minPointSqDis2)
                            {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--)
                    {
                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5*downsampleRate)
                        {
                            break;
                        }

                        pointSqDis = pointDistance(laserCloudCornerLast->points[j], pointSel);

                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan)
                        {
                            if (pointSqDis < minPointSqDis2)
                            {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                            }
                        }
                    }
                }
                
                if (minPointInd2 >= 0) // resize default to zero
                {
                    tripod1 = laserCloudCornerLast->points[closestPointInd];
                    tripod2 = laserCloudCornerLast->points[minPointInd2];
                    Eigen::Vector3d curr_point(laserCloudCornerSharp->points[i].x, // before projection
                                                laserCloudCornerSharp->points[i].y,
                                                laserCloudCornerSharp->points[i].z);
                    Eigen::Vector3d last_point_a(tripod1.x,tripod1.y,tripod1.z);                 
                    Eigen::Vector3d last_point_b(tripod2.x,tripod2.y,tripod2.z);

                    double s;
                    if (DISTORTION) // either before or after projection
                        s = (laserCloudCornerSharp->points[i].intensity - int(laserCloudCornerSharp->points[i].intensity)) / SCAN_PERIOD;
                    else
                        s = 1.0;
                    ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    corner_correspondence++;

                    if (opti_counter == opti_num - 1)
                    {
                        float x0 = pointSel.x;
                        float y0 = pointSel.y;
                        float z0 = pointSel.z;
                        float x1 = tripod1.x;
                        float y1 = tripod1.y;
                        float z1 = tripod1.z;
                        float x2 = tripod2.x;
                        float y2 = tripod2.y;
                        float z2 = tripod2.z;

                        float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                            * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                            + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                            * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                            + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                            * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                        float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                        odomRegistrationError.push_back(a012/l12);
                    }

                }
            }
            // surf points
            PointType tripod3,tripod4,tripod5;
            pointSearchInd.clear();
            pointSearchInd.clear();
            for (int i = 0; i < laserCloudSurfFlatNum; i++)
            {
                TransformToStart(&laserCloudSurfFlat->points[i], &pointSel);
                kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                if (pointSearchSqDis[0] < 25)
                {
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);

                    float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;
                    for (int j = closestPointInd + 1; j < laserCloudSurfFlatNum; j++)
                    {
                        if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5*downsampleRate)
                        {
                            break;
                        }
                        pointSqDis = pointDistance(laserCloudSurfLast->points[j], pointSel);

                         // choose two points across nearby scans
                         // if in the same or lower scan line (lower is not possible?)
                        if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan && pointSqDis < minPointSqDis2)
                        {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                        // if in the higher scan line
                        else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan && pointSqDis < minPointSqDis3)
                        {
                            minPointSqDis3 = pointSqDis;
                            minPointInd3 = j;
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--)
                    {
                        if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5*downsampleRate)
                        {
                            break;
                        }

                        pointSqDis = pointDistance(laserCloudSurfLast->points[j], pointSel);

                        // if in the same or higher scan line
                        if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan && pointSqDis < minPointSqDis2)
                        {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                        else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan && pointSqDis < minPointSqDis3)
                        {
                            // find nearer point
                            minPointSqDis3 = pointSqDis; 
                            minPointInd3 = j;
                        }
                    }
                }


                if (minPointInd2 >=0 && minPointInd3  >= 0)
                {
                    tripod3 = laserCloudSurfLast->points[closestPointInd];
                    tripod4 = laserCloudSurfLast->points[minPointInd2];
                    tripod5 = laserCloudSurfLast->points[minPointInd3];
                    Eigen::Vector3d curr_point(laserCloudSurfFlat->points[i].x,
                                                laserCloudSurfFlat->points[i].y,
                                                laserCloudSurfFlat->points[i].z);
                    Eigen::Vector3d last_point_a(tripod3.x,tripod3.y,tripod3.z);
                    Eigen::Vector3d last_point_b(tripod4.x,tripod4.y,tripod4.z);
                    Eigen::Vector3d last_point_c(tripod5.x,tripod5.y,tripod5.z);

                    double s;
                    if (DISTORTION)
                        s = (laserCloudSurfFlat->points[i].intensity - int(laserCloudSurfFlat->points[i].intensity)) / SCAN_PERIOD;
                    else
                        s = 1.0;
                    // find point a,b,c (to form a plane) in the last frame as the correspondence of curr_point
                    ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    plane_correspondence++;

                    if (opti_counter == opti_num -1)
                    {
                        
                        float pa = (tripod4.y - tripod3.y) * (tripod5.z - tripod3.z)
                        - (tripod5.y - tripod3.y) * (tripod4.z - tripod3.z);
                        float pb = (tripod4.z - tripod3.z) * (tripod5.x - tripod3.x)
                        - (tripod5.z - tripod3.z) * (tripod4.x - tripod3.x);
                        float pc = (tripod4.x - tripod3.x) * (tripod5.y - tripod3.y)
                        - (tripod5.x - tripod3.x) * (tripod4.y - tripod3.y);
                        float pd = -(pa * tripod3.x + pb * tripod3.y + pc * tripod3.z);
                        float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd; //Eq. (3)
                        odomRegistrationError.push_back(fabs(pd2));
                    }

                }
            }
            // printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
            // if ((corner_correspondence + plane_correspondence) < 50)
            // {
            //     printf("few correspondences: %d\n", corner_correspondence + plane_correspondence);
            // }

            TicToc t_solver;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            // printf("solver time %f ms \n", t_solver.toc());
        }
        // odom error
        regiError = accumulate(odomRegistrationError.begin(),odomRegistrationError.end(),0.0);
        regiError /= odomRegistrationError.size();
        vector<double> tmp;
        inlierRatio = (double)(corner_correspondence + plane_correspondence)/(laserCloudSurfFlatNum + laserCloudCornerSharpNum);
        tmp.push_back((double)cloudHeader.stamp.toSec());
        tmp.push_back(regiError);
        tmp.push_back(inlierRatio);
        odomErrorPerFrame.push_back(tmp);
        // ROS_INFO_STREAM("Odometry error: "<<regiError); 
        // if (regiError > 1.0 || inlierRatio < 0.7)
        // {
        //     t_w_curr = t_w_curr + q_w_curr * t_backup; 
        //     q_w_curr = q_w_curr * q_backup;
        //     t_last_curr = t_backup; // use zero guess for next matching
        //     q_last_curr = q_backup;
        // }
        // else
        // {
            // printf("optimization twice time %f \n", t_opt.toc());
            t_w_curr = t_w_curr + q_w_curr * t_last_curr; 
            q_w_curr = q_w_curr * q_last_curr;
        // }
        


    }

    void systemInitialization(){
        pcl::copyPointCloud(*laserCloudCornerCur,*laserCloudCornerLast);
        pcl::copyPointCloud(*laserCloudSurfCur,*laserCloudSurfLast);

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        systemInitedLM = true;
    }


    void publishOdometry(){

        // publish laser odometry
        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = q_w_curr.x();
        laserOdometry.pose.pose.orientation.y = q_w_curr.y();
        laserOdometry.pose.pose.orientation.z = q_w_curr.z();
        laserOdometry.pose.pose.orientation.w = q_w_curr.w();
        laserOdometry.pose.pose.position.x = t_w_curr.x();
        laserOdometry.pose.pose.position.y = t_w_curr.y();
        laserOdometry.pose.pose.position.z = t_w_curr.z();
        laserOdometry.twist.twist.linear.x = regiError; // borrow space for odom error
        pubLidarOdometry.publish(laserOdometry);

        // for rviz 
        geometry_msgs::PoseStamped pose_stamped; // notice here it is not nav_msgs::Odometry!
        pose_stamped.pose = laserOdometry.pose.pose;
        pose_stamped.header.stamp = cloudHeader.stamp;
        pose_stamped.header.frame_id = mapFrame;

        odometryPath.poses.push_back(pose_stamped);
        // Path message also needs stamp and frame id
        odometryPath.header.stamp = cloudHeader.stamp;
        odometryPath.header.frame_id = mapFrame;
        pubLidarPath.publish(odometryPath);
    }

    void laserCloudInfoHandler(const roll::cloud_infoConstPtr& msgIn)
    {
        std::lock_guard<std::mutex> lock(mtx);

        TicToc odometry;
        cloudHeader = msgIn->header;

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerCur);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfCur);
        pcl::fromROSMsg(msgIn->cloud_corner_sharp,  *laserCloudCornerSharp);
        pcl::fromROSMsg(msgIn->cloud_surface_flat, *laserCloudSurfFlat);
        if (laserCloudCornerCur->points.empty() || laserCloudSurfCur->points.empty() ||
                laserCloudCornerSharp->points.empty() || laserCloudSurfCur->points.empty())
        {
            ROS_WARN("lidar odometry: an empty frame?");
            return; // nclt20120108!
        }

        if ((int)laserCloudCornerCur->size() < edgeFeatureMinValidNum || (int)laserCloudSurfCur->size() <surfFeatureMinValidNum) // corner should be at least 20*6*32
        {
            ROS_WARN("lidar odometry: a degraded frame?");
            return; // nclt20120115 ~@4100!
        }
        
        if (!systemInitedLM) 
        {
            systemInitedLM = true;
        }
        else
        {
            laserCloudCornerSharpNum = laserCloudCornerSharp->size();
            laserCloudSurfFlatNum = laserCloudSurfFlat->size();

            // ROS_INFO_STREAM("Before sampling corner, surf: "<<laserCloudCornerCur->size()<<" "<<laserCloudSurfCur->size());
            // ROS_INFO_STREAM("After sampling corner, surf: "<<laserCloudCornerSharpNum<<" "<<laserCloudSurfFlatNum);
            optimization();
        }

        publishOdometry();
        
        // this way also works!
        pcl::copyPointCloud(*laserCloudCornerCur,*laserCloudCornerLast);
        pcl::copyPointCloud(*laserCloudSurfCur,*laserCloudSurfLast);
        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());
        // TicToc kdSet;
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);



        float odomTime = odometry.toc();
        if (odomTime > 100) {
            ROS_WARN("It takes %f ms to get odometry",odomTime);
        }
        odometryTimeVec.push_back(odomTime);
        // ROS_INFO_STREAM("Setting up kdtree takes: "<<kdSet.toc()<<" ms"); // < 2ms
        // ROS_INFO_STREAM("Odometry takes "<<odometryTimeVec[odometryTimeVec.size()-1]<<" ms"); // ~20ms for 16 line lidar
    }

};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "roll");
    lidarOdometry LO;   
    ROS_INFO("\033[1;32m----> Lidar Odometry Started.\033[0m");
    ros::spin();
    int sizeO = LO.odometryTimeVec.size();
    float sum = accumulate(LO.odometryTimeVec.begin(), LO.odometryTimeVec.end(),0);
    cout<<"Average time consumed by odometry is : "<<(double)sum/sizeO<<"ms"<<endl;
    
    if (LO.saveMatchingError) 
    {
        // saving odometry error
        string fileName = LO.saveMapDirectory + "/odomError.txt";
        ofstream odomErrorFile(fileName);
        odomErrorFile.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
        odomErrorFile.precision(6); // 固定小数位6
        if(!odomErrorFile.is_open())
        {
            cout<<"Cannot open "<<fileName<<endl;
            return false;
        }
        for (int i=0; i<(int)LO.odomErrorPerFrame.size();i++)
        {
            vector<double> tmp = LO.odomErrorPerFrame[i];
            odomErrorFile<<" "<<tmp[0]<<" " <<tmp[1]<<" "<<tmp[2]<<"\n";
        }
        odomErrorFile.close();
        cout<<"Done saving odometry error file!"<<endl;
    }
    
    return 0;
}
