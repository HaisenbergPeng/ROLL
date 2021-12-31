// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


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

#include "utility.h"
#include "kloam/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

using std::atan2;
using std::cos;
using std::sin;

double scanPeriod = 0.1;

class scanRegistration : public ParamServer
{

private:
    ros::NodeHandle nh;
    ros::Subscriber subLidarCloudInfo;

    ros::Publisher pubRawPoints;
    ros::Publisher pubProjPoints;
    
    ros::Publisher pubLidarCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;
    ros::Publisher pubSharpCornerPoints;
    ros::Publisher pubFlatSurfacePoints;

    pcl::PointCloud<PointType>::Ptr surfaceCloud2;

    ros::Publisher pubSurfacePoints2;

    pcl::VoxelGrid<PointType> downSizeFilter;

    vector<smoothness_t> cloudSmoothness;
    vector<float> cloudCurvature;
    vector<int> cloudNeighborPicked;
    vector<int> cloudLabel;

    // vector<pcl::PointCloud<PointType>> lidarCloudScans;

public:
    scanRegistration()
    {
        subLidarCloudInfo = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &scanRegistration::lidarCloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubLidarCloudInfo = nh.advertise<kloam::cloud_info> ("/kloam/feature/cloud_info", 1);
        pubRawPoints = nh.advertise<sensor_msgs::PointCloud2>("/kloam/feature/cloud_raw", 1);
        pubProjPoints = nh.advertise<sensor_msgs::PointCloud2>("/kloam/feature/cloud_projected", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("/kloam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("/kloam/feature/cloud_surface", 1);
        pubSurfacePoints2 = nh.advertise<sensor_msgs::PointCloud2>("/kloam/feature/cloud_surface2", 1);
        pubFlatSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("/kloam/feature/cloud_surface_flat", 1);
        pubSharpCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("/kloam/feature/cloud_corner_sharp", 1);
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        // to avoid overflow (sometimes points in one frame can be a lot)
        cloudCurvature.resize(N_SCAN*Horizon_SCAN*10);
        cloudNeighborPicked.resize(N_SCAN*Horizon_SCAN*10);
        cloudLabel.resize(N_SCAN*Horizon_SCAN*10);
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN*10);
    }

    // very important in outdoor SLAM
    // in nclt 20120202 loc run, it reduces reprojection error
    void markBadPoints(pcl::PointCloud<PointType>::Ptr cloudIn)
    {
        int cloudSize = cloudIn->size();
          for (int i = 5; i < cloudSize - 6; i++) 
          {
              
            float diffX = cloudIn->points[i + 1].x - cloudIn->points[i].x;
            float diffY = cloudIn->points[i + 1].y - cloudIn->points[i].y;
            float diffZ = cloudIn->points[i + 1].z - cloudIn->points[i].z;
            
            float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
            // 0.2 horizontal angle accuracy, so dist > 0.1/(0.2/57.3) = 28.65 m
            if (diff > lidarMinRange) 
            {
                float depth1 = sqrt(cloudIn->points[i].x * cloudIn->points[i].x + 
                                cloudIn->points[i].y * cloudIn->points[i].y +
                                cloudIn->points[i].z * cloudIn->points[i].z);

                
                float depth2 = sqrt(cloudIn->points[i + 1].x * cloudIn->points[i + 1].x + 
                                cloudIn->points[i + 1].y * cloudIn->points[i + 1].y +
                                cloudIn->points[i + 1].z * cloudIn->points[i + 1].z);

               
                if (depth1 > depth2) {
                    diffX = cloudIn->points[i + 1].x - cloudIn->points[i].x * depth2 / depth1;
                    diffY = cloudIn->points[i + 1].y - cloudIn->points[i].y * depth2 / depth1;
                    diffZ = cloudIn->points[i + 1].z - cloudIn->points[i].z * depth2 / depth1;

                    
                    if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {
                        
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                    }
                } else {
                    diffX = cloudIn->points[i + 1].x * depth1 / depth2 - cloudIn->points[i].x;
                    diffY = cloudIn->points[i + 1].y * depth1 / depth2 - cloudIn->points[i].y;
                    diffZ = cloudIn->points[i + 1].z * depth1 / depth2 - cloudIn->points[i].z;

                    if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                    }
                }
            }

            float diffX2 = cloudIn->points[i].x - cloudIn->points[i - 1].x;
            float diffY2 = cloudIn->points[i].y - cloudIn->points[i - 1].y;
            float diffZ2 = cloudIn->points[i].z - cloudIn->points[i - 1].z;

            float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;
            float dis = cloudIn->points[i].x * cloudIn->points[i].x
                    + cloudIn->points[i].y * cloudIn->points[i].y
                    + cloudIn->points[i].z * cloudIn->points[i].z;
            if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) 
            {
                cloudNeighborPicked[i] = 1;
            }
        }
    }

    void lidarCloudHandler(const sensor_msgs::PointCloud2ConstPtr &lidarCloudMsg)
    {

        kloam::cloud_info cloudInfo;

        TicToc t_whole;
        TicToc t_prepare;
        std::vector<int> scanStartInd(N_SCAN, 0);
        std::vector<int> scanEndInd(N_SCAN, 0);

        ros::Time lidarMsgStamp = lidarCloudMsg->header.stamp;
        pcl::PointCloud<PointType>::Ptr lidarCloudIn(new pcl::PointCloud<PointType>());
        
        pcl::fromROSMsg(*lidarCloudMsg, *lidarCloudIn);
            
        publishCloud(&pubRawPoints,  lidarCloudIn,  lidarMsgStamp, lidarFrame);
    
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*lidarCloudIn, *lidarCloudIn, indices);

        cloudInfo.cloud_raw = *lidarCloudMsg;


        int cloudSize = lidarCloudIn->points.size();

        if (cloudSize < 1000)
        {
            ROS_WARN("Empty cloud or points too few");
            return;
        } 
        if (cloudSize > 2*N_SCAN*Horizon_SCAN)
        {
            // ROS_WARN("Points too many");
            // pcl::io::savePCDFileBinary(saveMapDirectory + "/big_cloud.pcd", *lidarCloudIn);
        }
        float startOri = -atan2(lidarCloudIn->points[0].y, lidarCloudIn->points[0].x);
        float endOri = -atan2(lidarCloudIn->points[cloudSize - 1].y,
                            lidarCloudIn->points[cloudSize - 1].x) +
                    2 * M_PI;

        if (endOri - startOri > 3 * M_PI)
        {
            endOri -= 2 * M_PI;
        }
        else if (endOri - startOri < M_PI)
        {
            endOri += 2 * M_PI;
        }


        bool halfPassed = false;
        int count = cloudSize;
        PointType point;
        std::vector<pcl::PointCloud<PointType>> lidarCloudScans(N_SCAN);
        cv::Mat rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));;
        
        for (int i = 0; i < cloudSize; i++)
        {
            point.x = lidarCloudIn->points[i].x;
            point.y = lidarCloudIn->points[i].y;
            point.z = lidarCloudIn->points[i].z;

            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            if (N_SCAN == 16)
            {
                scanID = int((angle + 15) / 2 + 0.5);
                if (scanID > (N_SCAN - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (N_SCAN == 32)
            {
                scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                if (scanID > (N_SCAN - 1) || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else if (N_SCAN == 64)
            {   
                if (angle >= -8.83)
                    scanID = int((2 - angle) * 3.0 + 0.5);
                else
                    scanID = N_SCAN / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                // use [0 50]  > 50 remove outlies 
                if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                {
                    count--;
                    continue;
                }
            }
            else
            {
                printf("wrong scan number\n");
                ROS_BREAK();
            }


            float ori = -atan2(point.y, point.x);
            if (!halfPassed)
            { 
                if (ori < startOri - M_PI / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > startOri + M_PI * 3 / 2)
                {
                    ori -= 2 * M_PI;
                }

                if (ori - startOri > M_PI)
                {
                    halfPassed = true;
                }
            }
            else
            {
                ori += 2 * M_PI;
                if (ori < endOri - M_PI * 3 / 2)
                {
                    ori += 2 * M_PI;
                }
                else if (ori > endOri + M_PI / 2)
                {
                    ori -= 2 * M_PI;
                }
            }
        
                float relTime = (ori - startOri) / (endOri - startOri);
                point.intensity = scanID + scanPeriod * relTime;
                lidarCloudScans[scanID].push_back(point); 

        }
        // printf("Before projection, points size: %d \n", cloudSize);

        pcl::PointCloud<PointType>::Ptr lidarCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < N_SCAN; i++)
        { 
            scanStartInd[i] = lidarCloud->size() + 5;
            *lidarCloud += lidarCloudScans[i];
            scanEndInd[i] = lidarCloud->size() - 6;
        }

        cloudSize = lidarCloud->size();
        // cout<<"After projection, point size: "<<lidarCloud->size()<<endl;
        // printf("prepare time %f \n", t_prepare.toc());

        for (int i = 5; i < cloudSize - 5; i++)
        { 
            float diffX = lidarCloud->points[i - 5].x + lidarCloud->points[i - 4].x + lidarCloud->points[i - 3].x + lidarCloud->points[i - 2].x + lidarCloud->points[i - 1].x - 10 * lidarCloud->points[i].x + lidarCloud->points[i + 1].x + lidarCloud->points[i + 2].x + lidarCloud->points[i + 3].x + lidarCloud->points[i + 4].x + lidarCloud->points[i + 5].x;
            float diffY = lidarCloud->points[i - 5].y + lidarCloud->points[i - 4].y + lidarCloud->points[i - 3].y + lidarCloud->points[i - 2].y + lidarCloud->points[i - 1].y - 10 * lidarCloud->points[i].y + lidarCloud->points[i + 1].y + lidarCloud->points[i + 2].y + lidarCloud->points[i + 3].y + lidarCloud->points[i + 4].y + lidarCloud->points[i + 5].y;
            float diffZ = lidarCloud->points[i - 5].z + lidarCloud->points[i - 4].z + lidarCloud->points[i - 3].z + lidarCloud->points[i - 2].z + lidarCloud->points[i - 1].z - 10 * lidarCloud->points[i].z + lidarCloud->points[i + 1].z + lidarCloud->points[i + 2].z + lidarCloud->points[i + 3].z + lidarCloud->points[i + 4].z + lidarCloud->points[i + 5].z;

            cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
            cloudSmoothness[i].ind = i;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
        }
        // cout<<"smoothness calculated"<<endl;

        markBadPoints(lidarCloud);
        // cout<<"After removing bad points, point size: "<<lidarCloud->size()<<endl;
        TicToc t_pts;

        pcl::PointCloud<PointType>::Ptr cornerCloudSharp(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr  cornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr  surfaceCloudFlat(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr  surfaceCloud(new pcl::PointCloud<PointType>());


        float t_q_sort = 0;
        float t_filter = 0;
        for (int i = 0; i < N_SCAN; i++)
        {
            
            if ( i % downsampleRate != 0) continue;
            if( scanEndInd[i] - scanStartInd[i] < 6)
                continue;
            pcl::PointCloud<PointType>::Ptr surfaceCloudTmp(new pcl::PointCloud<PointType>);
            for (int j = 0; j < 6; j++)
            {
                int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
                int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

                TicToc t_tmp;
                std::sort (cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep + 1, by_value());
                t_q_sort += t_tmp.toc();

                int largestPickedNum = 0;
                
                for (int k = ep; k >= sp; k--)// cout<<"start filtering"<<endl;
                {
                    int ind = cloudSmoothness[k].ind; 

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 2)
                        {                        
                            cloudLabel[ind] = 2;
                            cornerCloudSharp->push_back(lidarCloud->points[ind]);
                            cornerCloud->push_back(lidarCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 20)
                        {                        
                            cloudLabel[ind] = 1; 
                            cornerCloud->push_back(lidarCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; 

                        for (int l = 1; l <= 5; l++)
                        {
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l - 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l - 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l + 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l + 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;

                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1; 
                        surfaceCloudFlat->push_back(lidarCloud->points[ind]);
                        smallestPickedNum++;
                        if (smallestPickedNum >= 4)
                        { 
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++)
                        { 
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l - 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l - 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l - 1].z;

                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }


                            cloudNeighborPicked[ind + l] = 1;

                        }
                        
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = lidarCloud->points[ind + l].x - lidarCloud->points[ind + l + 1].x;
                            float diffY = lidarCloud->points[ind + l].y - lidarCloud->points[ind + l + 1].y;
                            float diffZ = lidarCloud->points[ind + l].z - lidarCloud->points[ind + l + 1].z;

                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                            {
                                break;
                            }

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfaceCloudTmp->push_back(lidarCloud->points[k]);
                    }
                }
            }
            TicToc tmp;
            
            pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());
            
            downSizeFilter.setInputCloud(surfaceCloudTmp);
            downSizeFilter.filter(*surfaceCloudScanDS);
            
            t_filter += tmp.toc();
            *surfaceCloud += *surfaceCloudScanDS;
            
        }
        // printf("sort q time %f \n", t_q_sort);
        // printf("seperate points time %f \n", t_pts.toc());
        // printf("filter points time %f \n", t_filter);
        publishCloud(&pubProjPoints,  lidarCloud,  lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_corner_sharp  = publishCloud(&pubSharpCornerPoints,  cornerCloudSharp,  lidarMsgStamp, lidarFrame);
        cloudInfo.cloud_surface_flat = publishCloud(&pubFlatSurfacePoints, surfaceCloudFlat, lidarMsgStamp, lidarFrame);
        cloudInfo.header = lidarCloudMsg->header;
        // cout<<"finish"<<endl;
        pubLidarCloudInfo.publish(cloudInfo);

        // printf("scan registration time %f ms *************\n", t_whole.toc()); // ~ 40 ms
        if(t_whole.toc() > 100)
            ROS_WARN("scan registration process over 100ms");
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");

    scanRegistration SR;
    ROS_INFO("\033[1;32m----> Lidar Scan Registration Started.\033[0m");
    ros::spin();

    return 0;
}
