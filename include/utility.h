#pragma once
#ifndef _UTILITY_H_
#define _UTILITY_H_

#include"tic_toc.h"
#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include<pcl/visualization/pcl_visualizer.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include<cassert>
#include <utility>
#include <cstdlib>
#include <memory>
#include <numeric>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, LIVOX};

class ParamServer
{
public:
    // gps 
    double alti0;
    double lati0;
    double longi0;

    int optIteration;
    ros::NodeHandle nh;

    std::string robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;
    string gtTopic;

    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    bool savePose;
    bool saveKeyframeMap;
    bool saveRawCloud;
    bool mapUpdateEnabled;
    bool saveMatchingError;
    string saveMapDirectory;
    string saveKeyframeMapDirectory;
    string loadKeyframeMapDirectory;

    bool generateVocab;

    bool localizationMode;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    vector<double> initialGuess;
    
    
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    // temporary mapping
    float starttemporaryMappingDistThre; // key poses sparsified by 2.0 m
    float inlierThreshold; // if no serious map out-of-date, it suffice 99% of the time
    float startTemporaryMappingInlierRatioThre; 
    float exitTemporaryMappingInlierRatioThre; 
    int slidingWindowSize;

    ParamServer()
    {
        nh.param<double>("kloam/lati0", lati0, 0.0);
        nh.param<double>("kloam/longi0", longi0, 0.0);
        nh.param<double>("kloam/alti0", alti0, 0.0);

        nh.param<float>("kloam/starttemporaryMappingDistThre", starttemporaryMappingDistThre, 20.0);
        nh.param<float>("kloam/inlierThreshold", inlierThreshold, 0.2);
        nh.param<float>("kloam/startTemporaryMappingInlierRatioThre", startTemporaryMappingInlierRatioThre, 0.4);
        nh.param<float>("kloam/exitTemporaryMappingInlierRatioThre", exitTemporaryMappingInlierRatioThre, 0.4);
        nh.param<int>("kloam/slidingWindowSize", slidingWindowSize, 30);

        nh.param<std::string>("/robot_id", robot_id, "roboat");
        nh.param<int>("kloam/optIteration", optIteration,30);
        nh.param<std::string>("kloam/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("kloam/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("kloam/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("kloam/gpsTopic", gpsTopic, "fix");
        nh.param<std::string>("kloam/gtTopic", gtTopic, "ground_truth");

        nh.param<std::string>("kloam/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("kloam/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("kloam/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("kloam/mapFrame", mapFrame, "map");

        nh.param<bool>("kloam/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("kloam/useGpsElevation", useGpsElevation, false);
        nh.param<float>("kloam/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("kloam/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>("kloam/saveMatchingError", saveMatchingError, true);
        nh.param<bool>("kloam/savePCD", savePCD, false);
        nh.param<bool>("kloam/savePose", savePose, false);
        nh.param<bool>("kloam/saveKeyframeMap", saveKeyframeMap, false);
        nh.param<bool>("kloam/saveRawCloud", saveRawCloud, false);
        nh.param<bool>("kloam/localizationMode", localizationMode, false);
        nh.param<bool>("kloam/mapUpdateEnabled", mapUpdateEnabled, false);

        
        nh.param<std::string>("kloam/saveMapDirectory", saveMapDirectory, "/Downloads/LOAM/");
        nh.param<std::string>("kloam/loadKeyframeMapDirectory", loadKeyframeMapDirectory, "/Downloads/LOAM/");
        nh.param<std::string>("kloam/saveKeyframeMapDirectory", saveKeyframeMapDirectory, "/Downloads/LOAM/");        


        std::string sensorStr;
        nh.param<std::string>("kloam/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
            ros::shutdown();
        }

        nh.param<int>("kloam/N_SCAN", N_SCAN, 16);
        nh.param<int>("kloam/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("kloam/downsampleRate", downsampleRate, 1);
        nh.param<float>("kloam/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("kloam/lidarMaxRange", lidarMaxRange, 1000.0);

        nh.param<vector<double>>("kloam/initialGuess", initialGuess, vector<double>(6, 0));

        nh.param<float>("kloam/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("kloam/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("kloam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("kloam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("kloam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("kloam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("kloam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("kloam/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("kloam/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("kloam/numberOfCores", numberOfCores, 2);

        nh.param<float>("kloam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("kloam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("kloam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("kloam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>("kloam/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("kloam/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("kloam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("kloam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("kloam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("kloam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("kloam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("kloam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("kloam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("kloam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);
        usleep(100);
    }
};


sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0) // this is important since publishing actually takes quite some time
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}


void printTrans(std::string message, vector<double> transformIn){
    if (transformIn.size() != 6) return;
    cout<<message<<transformIn[0]<<" "<<transformIn[1]<<" "
        <<transformIn[2]<<" "<<transformIn[3]<<" "<<transformIn[4]<<" "<<transformIn[5];
    return;
}
// returning c array is not easy to pull off, just forget it
void Affine3f2Trans(Eigen::Affine3f t,float transformOut[6])
{
    pcl::getTranslationAndEulerAngles(t,transformOut[3], transformOut[4], transformOut[5], transformOut[0], transformOut[1], transformOut[2]);
}

Eigen::Affine3f trans2Affine3f(float transformIn[6])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

void printTrans(std::string message, const float transformIn[]){
    ROS_INFO_STREAM(message<<transformIn[0]<<" "<<transformIn[1]<<" "
        <<transformIn[2]<<" "<<transformIn[3]<<" "<<transformIn[4]<<" "<<transformIn[5]);
    return;
}

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}

#endif
