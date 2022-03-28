/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <vector>
#include <map>
#include <iostream>
#include <mutex>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include "tic_toc.h"
using namespace std;

class GlobalOptimization
{
public:
	GlobalOptimization(int maxNo);
	~GlobalOptimization();
	void setTgl(Eigen::Matrix4d mat);
	void getTgl(Eigen::Matrix4d &tgl);
	void affine2qt(const Eigen::Matrix4d aff, Eigen::Vector3d &locP, Eigen::Quaterniond &locQ);
	// void inputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy);
	void inputOdom(double t, Eigen::Matrix4d affine);
	void inputGlobalLocPose(double t, Eigen::Matrix4d affine, double t_error, double q_error);
	void getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ);
	void getGlobalAffine(Eigen::Affine3f &Tml);

	void resetOptimization(Eigen::Matrix4d Tgl);
	nav_msgs::Path global_path;

	bool isInitialized;
	bool reInitialize;
private:
	// void GPS2XYZ(double latitude, double longitude, double altitude, double* xyz);
	void optimize();
	void updateGlobalPath();

	// format t, tx,ty,tz,qw,qx,qy,qz
	map<double, vector<double>> localPoseMap;
	map<double, vector<double>> globalPoseMap; // only synchronized gps is used
	map<double, vector<double>> GPSPositionMap;
	map<double, vector<double>> globalLocPoseMap;
	bool initGPS;
	bool newGPS;
	bool newGlobalLocPose;
	// GeographicLib::LocalCartesian geoConverter;
	std::mutex mPoseMap;
	Eigen::Matrix4d WGlobal_T_WLocal;
	Eigen::Vector3d lastP;
	Eigen::Quaterniond lastQ;
	int maxFrameNum;

	std::thread threadOpt;

	// for acc jump identification
	vector<Eigen::Matrix4d> poseVec;
	Eigen::Matrix4d Tacc;
	bool resetGlobalPoses;
	Eigen::Matrix4d backupTgl;
	// double stdx,meanx,stdy,meany,stdz,meanz;
};