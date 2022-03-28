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

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            // options.linear_solver_type = ceres::DENSE_QR;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.5);
            ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();

            //add param
            mPoseMap.lock();

            int length = localPoseMap.size();
            // w^t_i   w^q_i
            double t_array[length][3];
            double q_array[length][4];
            map<double, vector<double>>::iterator iter;

            // cannot use global matching poses for initials: too sparse
            iter = globalPoseMap.begin();  //using odomTOmap value for initial guess 
            for (int i = 0; i < length; i++, iter++)
            {
                t_array[i][0] = iter->second[0];
                t_array[i][1] = iter->second[1];
                t_array[i][2] = iter->second[2];
                q_array[i][0] = iter->second[3];
                q_array[i][1] = iter->second[4];
                q_array[i][2] = iter->second[5];
                q_array[i][3] = iter->second[6];
                // cout<<"q_array[i] "<<q_array[i][0]<<" "<<q_array[i][1]<<" "<<q_array[i][2]<<" "<<q_array[i][3]<<endl;
                problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);
            }

            // cout<<" pose no. before opt "<< length<<endl;
            map<double, vector<double>>::iterator iterVIO, iterVIONext, iterGlobalLoc;
            int i = 0;
            int found  = 0;
            // int outlierNO = 0;
            for (iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); iterVIO++, i++)
            {
                //vio factor
                iterVIONext = iterVIO;
                iterVIONext++;
                if(iterVIONext != localPoseMap.end())
                {
                    Eigen::Matrix4d wTi = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d wTj = Eigen::Matrix4d::Identity();
                    wTi.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4], 
                                                               iterVIO->second[5], iterVIO->second[6]).toRotationMatrix();
                    wTi.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                    wTj.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIONext->second[3], iterVIONext->second[4], 
                                                               iterVIONext->second[5], iterVIONext->second[6]).toRotationMatrix();
                    wTj.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIONext->second[0], iterVIONext->second[1], iterVIONext->second[2]);
                    Eigen::Matrix4d iTj = wTi.inverse() * wTj;
                    Eigen::Quaterniond iQj;
                    iQj = iTj.block<3, 3>(0, 0);
                    Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);

                    ceres::CostFunction* vio_function = RelativeRTError::Create(iPj.x(), iPj.y(), iPj.z(),
                                                                                iQj.w(), iQj.x(), iQj.y(), iQj.z(),
                                                                                0.1, 0.01);
                    problem.AddResidualBlock(vio_function, NULL, q_array[i], t_array[i], q_array[i+1], t_array[i+1]);

                }
                
                double t = iterVIO->first;
                iterGlobalLoc = globalLocPoseMap.find(t); // synchronized global loc pose
                if (iterGlobalLoc != globalLocPoseMap.end())
                {
                        
                    // cout<<"found corresponding global loc pose "<<endl;
                    ceres::CostFunction* global_loc_function = globalTError::Create(iterGlobalLoc->second[0], iterGlobalLoc->second[1], 
                                                                       iterGlobalLoc->second[2], iterGlobalLoc->second[3],
                                                                       iterGlobalLoc->second[4],iterGlobalLoc->second[5],iterGlobalLoc->second[6],
                                                                       iterGlobalLoc->second[7],iterGlobalLoc->second[8]);
                    
                    problem.AddResidualBlock(global_loc_function, loss_function, q_array[i], t_array[i]);

                    // ceres::CostFunction* global_loc_function = TError::Create(iterGlobalLoc->second[0], iterGlobalLoc->second[1], 
                    //                                                    iterGlobalLoc->second[2], iterGlobalLoc->second[7]);
                    
                    // problem.AddResidualBlock(global_loc_function, loss_function, t_array[i]);
                    found++;
                    Eigen::Matrix4d local = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d global = Eigen::Matrix4d::Identity();
                    global.block<3, 3>(0, 0) = Eigen::Quaterniond(iterGlobalLoc->second[3], iterGlobalLoc->second[4], 
                                                                        iterGlobalLoc->second[5], iterGlobalLoc->second[6]).toRotationMatrix();
                    global.block<3, 1>(0, 3) = Eigen::Vector3d(iterGlobalLoc->second[0], iterGlobalLoc->second[1], iterGlobalLoc->second[2]);
                    local.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4], 
                                                                        iterVIO->second[5], iterVIO->second[6]).toRotationMatrix();
                    local.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                    backupTgl = global*local.inverse(); // get the newest Tgl as backup
                }

            }

            
            // cout<<"found: "<<found<<endl;
            ceres::Solve(options, &problem, &summary);
            // std::cout << summary.BriefReport() << "\n";

            // update global pose
            iter = globalPoseMap.begin();

            Eigen::Matrix4d start;
            Eigen::Matrix4d end;
            Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity(); 
            Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
        
            for (int i = 0; i < length; i++, iter++)
            {
                // cout<<"i "<<i<<"length :"<<length<<endl;
                vector<double> globalPose{t_array[i][0], t_array[i][1], t_array[i][2],
                                        q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]};
                iter->second = globalPose;
                // only output the lastest
                // if(i == length - 1)
                // {
                //     Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity(); 
                //     Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
                //     double t = iter->first;
                //     WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4], 
                //                                                        localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
                //     WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                //     WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4], 
                //                                                         globalPose[5], globalPose[6]).toRotationMatrix();
                //     WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
                //     WGlobal_T_WLocal = WGPS_T_body * WVIO_T_body.inverse();
                // }

                double t = iter->first;
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


            // Tgl change too much, forfeit this optimization
            Eigen::Matrix4d TglDelta = start.inverse()*end;
            // cout<<"x y z: "<<TglDelta(0,3)<<" "<<TglDelta(1,3)<<" "<<TglDelta(2,3)<<endl;
            double deltaTransGL = sqrt(TglDelta(0,3)*TglDelta(0,3) +  TglDelta(1,3)*TglDelta(1,3) + TglDelta(2,3)*TglDelta(2,3) );

            // cout<<"Tgl change "<<deltaTransGL<<endl;
            // w. consistency check
            if ( deltaTransGL > 0.5)
            {
                resetOptimization(backupTgl);
            }
            // cout<<"optimization takes: "<<opt_time.toc()<<" ms"<<endl;
            mPoseMap.unlock();
        }
        // waiting for poses to be accumulated
        std::chrono::milliseconds dura(500);
        std::this_thread::sleep_for(dura);
    }
	return;
}

