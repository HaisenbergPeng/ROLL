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

// adapted for global loc
// 1. add init function;
// 2. add sliding-window opt.
// 3. add outlier detection

#include "globalOpt.h"
#include "Factors.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

GlobalOptimization::GlobalOptimization(int maxNo)
{
	initGPS = false;
    newGPS = false;
    newGlobalLocPose = false;
	WGlobal_T_WLocal = Eigen::Matrix4d::Identity();
    threadOpt = std::thread(&GlobalOptimization::optimize, this);
    maxFrameNum = maxNo;

    reInitialize = false;
    Tacc = Eigen::Matrix4d::Identity();
    backupTgl = Eigen::Matrix4d::Identity();
    
}
void GlobalOptimization::setTgl(Eigen::Matrix4d mat)
{
    WGlobal_T_WLocal = mat;
    backupTgl = mat;
}

GlobalOptimization::~GlobalOptimization()
{
    threadOpt.detach();
}

void GlobalOptimization::affine2qt(const Eigen::Matrix4d aff, Eigen::Vector3d &p, Eigen::Quaterniond &q)
{
    q = Eigen::Quaterniond(aff.block<3,3>(0,0));
    p = aff.block<3,1>(0,3);    
    // cout<<"q: "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<endl;
    return;
}

// here we update globally aligned odom pose asap
void GlobalOptimization::inputOdom(double t, Eigen::Matrix4d affine)
{
    Eigen::Vector3d OdomP;
    Eigen::Quaterniond OdomQ;
    affine2qt(affine,OdomP,OdomQ);

	mPoseMap.lock();
    vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(), 
    					     OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z()};
    localPoseMap[t] = localPose;

    // cout<<setiosflags(ios::fixed)<<setprecision(6)<<"lio: "<<t<<endl;

    Eigen::Quaterniond globalQ;
    globalQ = WGlobal_T_WLocal.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGlobal_T_WLocal.block<3, 3>(0, 0) * OdomP + WGlobal_T_WLocal.block<3, 1>(0, 3);
    vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                              globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose; 
    lastP = globalP;
    lastQ = globalQ;

    // sliding-window
    if ((int)localPoseMap.size() > maxFrameNum)
    {
        // special note: erase function of map is not the same as vector
        localPoseMap.erase((localPoseMap.begin())->first);
        globalPoseMap.erase((globalPoseMap.begin())->first);
    }

    mPoseMap.unlock();
}

void GlobalOptimization::getGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ)
{
    odomP = lastP;
    odomQ = lastQ;
}

void GlobalOptimization::getGlobalAffine(Eigen::Affine3f &Tml)
{
    // need to wait for the 1st opt. completion
    Eigen::Matrix4d tmp;
    tmp.block<3,3>(0,0) = lastQ.toRotationMatrix();
    tmp.block<3,1>(0,3) = lastP;
    Tml = tmp.cast<float>();
}

void GlobalOptimization::getTgl(Eigen::Matrix4d &tgl)
{
    tgl = WGlobal_T_WLocal;
}


void GlobalOptimization::inputGlobalLocPose(double t, Eigen::Matrix4d affine, double t_error, double q_error)
{
    mPoseMap.lock();
    Eigen::Vector3d locP;
    Eigen::Quaterniond locQ;
    affine2qt(affine,locP,locQ);
    vector<double> globalLocPose{locP.x(), locP.y(), locP.z(),
                              locQ.w(), locQ.x(), locQ.y(), locQ.z(), t_error, q_error};
    globalLocPoseMap[t] = globalLocPose;
    // vector<double> globalLocPose{locP.x(), locP.y(), locP.z(),t_error};
    // globalLocPoseMap[t] = globalLocPose; 

    // cout<<setiosflags(ios::fixed)<<setprecision(6)<<"global matching: "<<t<<endl;

    if (reInitialize == true && (int)globalLocPoseMap.size() < 10) 
    {
        mPoseMap.unlock();
        return;
    }
    newGlobalLocPose = true;
    
    mPoseMap.unlock();
}

void GlobalOptimization::resetOptimization(Eigen::Matrix4d Tgl)
{
    // cout<<"too much jump in Tgl change, forfeit estimate:  Tgl change "<<deltaTransGL<<endl;
    WGlobal_T_WLocal = Tgl;
    // while (globalLocPoseMap.empty() == false)
    // {
    //     // should be thread safe
    //     globalLocPoseMap.erase((globalLocPoseMap.begin())->first);
    // }
    // while (localPoseMap.empty() == false)
    // {
    //     // should be thread safe
    //     localPoseMap.erase((localPoseMap.begin())->first);
    //     globalPoseMap.erase((globalPoseMap.begin())->first);
    // }
    // OR
    globalLocPoseMap.clear();
    globalPoseMap.clear();
    localPoseMap.clear();
    reInitialize = true;

}

gtsam::Pose3 QT2gtsamPose(vector<double> qt)
{
    Eigen::Quaterniond q(qt[3],qt[4],qt[5],qt[6]);
    Eigen::Matrix3d qtMatrix = q.matrix();
    // here it returns (thetaZ,thetaY,thetaX)
    Eigen::Vector3d ypr = qtMatrix.eulerAngles(2,1,0);
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(ypr[2], ypr[1], ypr[0]), 
                                  gtsam::Point3(qt[0], qt[1], qt[2]));
}

void gtsamPose2Matrix4d(gtsam::Pose3 pose, Eigen::Matrix4d & mat)
{
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(pose.rotation().roll(),Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pose.rotation().pitch(),Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(pose.rotation().yaw(),Eigen::Vector3d::UnitZ())); 
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix = yawAngle*pitchAngle*rollAngle;
    mat.block<3,3>(0,0) = rotation_matrix;
    mat.block<3,1>(0,3) = Eigen::Vector3d(pose.translation().x(),pose.translation().y(),pose.translation().z());

}
void gtsamPose2Vector(gtsam::Pose3 pose, vector<double> &qt)
{
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(pose.rotation().roll(),Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(pose.rotation().pitch(),Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(pose.rotation().yaw(),Eigen::Vector3d::UnitZ())); 
    Eigen::Matrix3d rotation_matrix = (yawAngle*pitchAngle*rollAngle).matrix();
    Eigen::Quaterniond qd =  Eigen::Quaterniond(rotation_matrix);
    qt[0] = pose.translation().x();
    qt[1] = pose.translation().y();
    qt[2] = pose.translation().z();
    qt[3] = qd.w(); // are you sure?
    qt[4] = qd.x();
    qt[5] = qd.y();
    qt[6] = qd.z();
}
void GlobalOptimization::optimize()
{
    while(true)
    {
        if(newGlobalLocPose)
        {
            if (reInitialize == true) reInitialize = false;

            TicToc opt_time;
            newGlobalLocPose = false;
            // printf("global optimization using global localization and odometry\n");
            TicToc globalOptimizationTime;

            // gtsam
            NonlinearFactorGraph gtSAMgraphTM;
            Values initialEstimateTM;        
            ISAM2 *isamTM;
            Values isamCurrentEstimateTM;
            ISAM2Params parameters;
            parameters.relinearizeThreshold = 0.1;
            parameters.relinearizeSkip = 1;
            isamTM = new ISAM2(parameters);

            //add param
            mPoseMap.lock();

            int length = localPoseMap.size();
            // cout<<" pose no. before opt "<< length<<endl;
            map<double, vector<double>>::iterator iterIni,iterLIO, iterLIOnext, iterGlobalLoc;
            iterIni = globalPoseMap.begin();  //using odomTOmap value for initial guess 
            int i = 0;
            int found  = 0;
            // int outlierNO = 0;
            for (iterLIO = localPoseMap.begin(); iterLIO != localPoseMap.end(); iterLIO++, i++,iterIni++)
            {
                //vio factor
                // cout<<"i lio pose: " <<i<<endl;
                iterLIOnext = iterLIO;
                iterLIOnext++;
                if(iterLIOnext != localPoseMap.end())
                {
                    noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) <<1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2 ).finished());
                    gtsam::Pose3 poseFrom = QT2gtsamPose(iterLIO->second);
                    gtsam::Pose3 poseTo   = QT2gtsamPose(iterLIOnext->second);
                    gtSAMgraphTM.add(BetweenFactor<Pose3>(i,i+1, poseFrom.between(poseTo), odometryNoise));
                }
   
                double t = iterLIO->first;
                iterGlobalLoc = globalLocPoseMap.find(t); // synchronized global loc pose
                if (iterGlobalLoc != globalLocPoseMap.end())
                {
                    gtsam::Pose3 poseGlobal = QT2gtsamPose(iterGlobalLoc->second);
                    // seeems to be overconfident
                    // noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1).finished()); // rad*rad, meter*meter
                    double tE = iterGlobalLoc->second[7];
                    double tQ = iterGlobalLoc->second[8];
                    noiseModel::Diagonal::shared_ptr corrNoise = noiseModel::Diagonal::Variances((Vector(6) << tQ*tQ,tQ*tQ,tQ*tQ,tE*tE,tE*tE,tE*tE).finished()); // rad*rad, meter*meter
                    gtSAMgraphTM.add(PriorFactor<Pose3>(i, poseGlobal, corrNoise));
                    found++;

                    // for CC
                    Eigen::Matrix4d local = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d global = Eigen::Matrix4d::Identity();
                    global.block<3, 3>(0, 0) = Eigen::Quaterniond(iterGlobalLoc->second[3], iterGlobalLoc->second[4], 
                                                                        iterGlobalLoc->second[5], iterGlobalLoc->second[6]).toRotationMatrix();
                    global.block<3, 1>(0, 3) = Eigen::Vector3d(iterGlobalLoc->second[0], iterGlobalLoc->second[1], iterGlobalLoc->second[2]);
                    local.block<3, 3>(0, 0) = Eigen::Quaterniond(iterLIO->second[3], iterLIO->second[4], 
                                                                        iterLIO->second[5], iterLIO->second[6]).toRotationMatrix();
                    local.block<3, 1>(0, 3) = Eigen::Vector3d(iterLIO->second[0], iterLIO->second[1], iterLIO->second[2]);
                    backupTgl = global*local.inverse(); // get the newest Tgl as backup
                }

                gtsam::Pose3 poseGuess = QT2gtsamPose(iterIni->second);
                initialEstimateTM.insert(i, poseGuess);
            }

            if (found == 0)
            {
                mPoseMap.unlock();
                continue;
            }

            // cout<<"found: "<<found<<endl; // why zero from the start???
            isamTM->update(gtSAMgraphTM, initialEstimateTM);
            isamTM->update();
            isamTM->update();
            gtSAMgraphTM.resize(0);
            initialEstimateTM.clear();

            isamCurrentEstimateTM = isamTM->calculateEstimate();

            // cout<<"Estimate size: "<<length<<endl;
            // update global pose
            iterIni = globalPoseMap.begin();

            Eigen::Matrix4d start;
            Eigen::Matrix4d end;
            Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity(); 
            Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
        
            for (int i = 0; i < length; i++, iterIni++)
            {
                // cout<<"i "<<i<<"length :"<<length<<endl;
                vector<double> globalPose(7,0);
                gtsamPose2Vector(isamCurrentEstimateTM.at<Pose3>(i),globalPose);
                iterIni->second = globalPose;

                double t = iterIni->first;
                WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4], 
                                                                    localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
                WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4], 
                                                                    globalPose[5], globalPose[6]).toRotationMatrix();
                WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);

                WGlobal_T_WLocal = WGPS_T_body * WVIO_T_body.inverse();

                if (i == 0 ) start = WGlobal_T_WLocal;
                if (i == length - 1 ) end = WGlobal_T_WLocal;
            }
            

            // cout<<"Tgl change "<<deltaTransGL<<endl;
            // w. consistency check

            // interesting: implementation in gtsam has no need for CC

            // Tgl change too much, forfeit this optimization
            // Eigen::Matrix4d TglDelta = start.inverse()*end;
            // cout<<"x y z: "<<TglDelta(0,3)<<" "<<TglDelta(1,3)<<" "<<TglDelta(2,3)<<endl;
            // double deltaTransGL = sqrt(TglDelta(0,3)*TglDelta(0,3) +  TglDelta(1,3)*TglDelta(1,3) + TglDelta(2,3)*TglDelta(2,3) );
            // if ( deltaTransGL > 0.5)
            // {
            //     cout<<"reset when deltaTgl = "<<deltaTransGL<<endl;
            //     resetOptimization(backupTgl);
            // }
            // cout<<"optimization takes: "<<opt_time.toc()<<" ms"<<endl; // gtsam implementation takes less than 10 ms

            mPoseMap.unlock();
        }
        // waiting for poses to be accumulated
        std::chrono::milliseconds dura(500);
        std::this_thread::sleep_for(dura);
    }
	return;
}

